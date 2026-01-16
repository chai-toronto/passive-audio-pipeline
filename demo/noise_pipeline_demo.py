import pandas as pd
import numpy as np
import os
from demo.audio import RemoveSilence, SileroVAD, Diarization

import uuid
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import noisereduce as nr
import opensmile
from core.interfaces import BaseFilter, BaseFeatureExtractor, BaseTransformation, BaseCharacteristicMeasure, BaseDistanceMeasure

import sys
import types

try:
    import visqol.pb2.visqol_config_pb2
    import visqol.pb2.similarity_result_pb2
    
    src = types.ModuleType('src')
    src.proto = types.ModuleType('src.proto')
    
    src.proto.visqol_config_pb2 = visqol.pb2.visqol_config_pb2
    src.proto.similarity_result_pb2 = visqol.pb2.similarity_result_pb2
    
    sys.modules['src'] = src
    sys.modules['src.proto'] = src.proto
    sys.modules['src.proto.visqol_config_pb2'] = src.proto.visqol_config_pb2
    sys.modules['src.proto.similarity_result_pb2'] = src.proto.similarity_result_pb2

except ImportError as e:
    print(f"VisQOL import failed: {e}")

class SilenceFilter(BaseFilter):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return RemoveSilence(df)

class VAD(BaseFilter):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return SileroVAD(df)

class PrimarySpeaker(BaseFilter):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        diariziation = Diarization()
        return diariziation.run(df)

class OpenSMILE(BaseFeatureExtractor):
    def __init__(self, feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals):
        self.feature_set = feature_set
        self.feature_level = feature_level
        self.smile = opensmile.Smile(
            feature_set=self.feature_set,
            feature_level=self.feature_level
        )

    def _process_row(self, row):
        try:
            path = row['filename']
            if not os.path.exists(path):
                return None

            sampling_rate, sig = wav.read(path)
            
            start_sample = int(row['start'])
            end_sample = int(row['end'])
            
            if start_sample < 0: start_sample = 0
            if end_sample > len(sig): end_sample = len(sig)
            if start_sample >= end_sample:
                return None

            data = sig[start_sample:end_sample]
            min_samples = int(sampling_rate * 0.5) 
            
            if len(data) < min_samples:
                pad_needed = min_samples - len(data)
                data = np.pad(data, (0, pad_needed), mode='constant', constant_values=0)

            if data.dtype != np.float32 and data.dtype != np.float64:
                data = data.astype(np.float32) / 32768.0

            features = self.smile.process_signal(data, sampling_rate=sampling_rate)
            features.reset_index(drop=True, inplace=True)
            return features
            
        except Exception as e:
            print(f"Error extracting OpenSMILE features for {row['filename']}: {e}")
            return None

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            res = self._process_row(row)
            if res is not None and not res.empty:
                results.append(res)
        
        if not results:
            return pd.DataFrame()
            
        features_df = pd.concat(results, ignore_index=True)
        return features_df

class NoiseReduce(BaseTransformation):
    def __init__(self, temp_dir="demo/temp_denoised"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def _denoise_and_save(self, row):
        try:
            path = row['filename']            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                return row
            sampling_rate, sig = wav.read(path)
            start = int(row['start'])
            end = int(row['end'])
            
            if start < 0:
                start = 0
            if end > len(sig):
                end = len(sig)

            data = sig[start:end]

            if len(data) == 0:
                return row

            clean_data = nr.reduce_noise(y=data, sr=sampling_rate, stationary=False, chunk_size=32000)

            unique_name = f"{uuid.uuid4()}.wav"
            new_path = os.path.join(self.temp_dir, unique_name)
            
            if sig.dtype == np.int16:
                wav.write(new_path, sampling_rate, clean_data.astype(np.int16))
            else:
                wav.write(new_path, sampling_rate, clean_data)

            new_row = row.copy()
            new_row['filename'] = new_path
            new_row['start'] = 0 
            new_row['end'] = len(clean_data)
            return new_row

        except Exception as e:
            print(f"Error denoising {path}: {e}")
            return row

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        results_df = df.apply(self._denoise_and_save, axis=1)     
        return results_df
    
class VisQOL(BaseCharacteristicMeasure):
    def __init__(self, mode='speech'):
        self.mode = mode
        
        self.refs_paths = [
            "demo/reference_audio/male1_clean_16k.wav", "demo/reference_audio/male2_clean_16k.wav",
            "demo/reference_audio/male3_clean_16k.wav", "demo/reference_audio/male4_clean_16k.wav",
            "demo/reference_audio/female1_clean_16k.wav", "demo/reference_audio/female2_clean_16k.wav",
            "demo/reference_audio/female3_clean_16k.wav", "demo/reference_audio/female4_clean_16k.wav"
        ]

        self.config = self._initialize_config(mode)
        self.ref_datas = self._load_refs()

    def _initialize_config(self, mode):
        
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        config = visqol_config_pb2.VisqolConfig()

        if mode == "audio":
            config.audio.sample_rate = 48000
            config.options.use_speech_scoring = False
            svr_model_path = "libsvm_nu_svr_model.txt"
        else:
            config.audio.sample_rate = 16000
            config.options.use_speech_scoring = True
            svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"

        config.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
        )
        return config

    def _load_refs(self):
        loaded_refs = []
        for ref_path in self.refs_paths:
            if os.path.exists(ref_path):
                try:
                    fs, data = wav.read(ref_path)
                    loaded_refs.append(data.astype(float))
                except Exception as e:
                    print(f"Error loading ref {ref_path}: {e}")
            else:
                print(f"Warning: Reference file {ref_path} not found.")
        return loaded_refs

    def _calculate_row_moslqo(self, row):
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        try:
            path = row['filename']
            if not os.path.exists(path):
                return 0.0

            sr, sig = wav.read(path)
            
            start = int(row['start'])
            end = int(row['end'])
             
            if start < 0: start = 0
            if end > len(sig): end = len(sig)
            
            if start >= end:
                return 0.0

            data = sig[start:end].astype(float)
            deg_d = data

            if len(deg_d) < 16000:
                pad_amt = 16000 - len(deg_d)
                deg_d = np.pad(deg_d, (0, pad_amt), 'constant')

            api = visqol_lib_py.VisqolApi()
            api.Create(self.config)

            best_moslqo = 0.0

            for ref_data in self.ref_datas:
                
                max_len = max(len(deg_d), len(ref_data))
                pad_ref = max_len - len(ref_data)
                pad_deg = max_len - len(deg_d)
                
                r_padded = np.pad(ref_data, (0, pad_ref), 'constant')
                d_padded = np.pad(deg_d, (0, pad_deg), 'constant')

                similarity = api.Measure(r_padded, d_padded)                
                score = getattr(similarity, 'moslqo', 0)
                if score > best_moslqo:
                    best_moslqo = score
            return best_moslqo

        except Exception as e:
            print(f"VisQOL Error for {row.get('filename', 'unknown')}: {e}")
            return 0.0

    def measure(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = df.apply(self._calculate_row_moslqo, axis=1)
        return pd.DataFrame(scores, columns=['moslqo'])
    
class PearsonCoefficient(BaseDistanceMeasure):
    def calculate(self, f1: pd.DataFrame, f2: pd.DataFrame) -> pd.Series:        
        if len(f1) != len(f2):
            print("Warning: Feature DataFrames have different lengths.")
            
        common_cols = list(set(f1.select_dtypes(include=np.number).columns) & 
                           set(f2.select_dtypes(include=np.number).columns))
        
        f1_sub = f1[common_cols]
        f2_sub = f2[common_cols]

        correlations = f1_sub.corrwith(f2_sub, axis=0)
        
        return correlations