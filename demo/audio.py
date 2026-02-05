import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy import signal
import torch
from pyannote.audio import Pipeline
import os
import librosa
from dotenv import load_dotenv

load_dotenv() # Load environment variables

def RemoveSilence(df, threshold=1000, moving_avg_window=0.5):
    output_rows = []

    for index, row in df.iterrows():
        fname = row['filename']
        date_val = row['date']
        timestamp = row['timestamp']
        
        try:
            fs, audio = wavfile.read(fname)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            b, a = a_weighting(fs)
            a_weighted_data = signal.lfilter(b, a, audio)

            squared_a_weighted = np.square(a_weighted_data)

            h = signal.firwin(numtaps=10, cutoff=40, fs=fs)
            lpf = signal.lfilter(h, 1.0, squared_a_weighted)

            window_samples = int(moving_avg_window * fs)
            averaged_lpf = moving_average(lpf, window_samples)

            if isinstance(averaged_lpf, pd.Series):
                averaged_lpf = averaged_lpf.values

            mask = np.copy(averaged_lpf)
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1

            padded_mask = np.concatenate(([0], mask, [0]))
            diff = np.diff(padded_mask.astype(int))
            
            start_indices = np.where(diff == 1)[0]
            end_indices = np.where(diff == -1)[0]
            
            for start, end in zip(start_indices, end_indices):
                output_rows.append({
                    'filename': fname,
                    'date': date_val,
                    'start': start / fs,
                    'end': end / fs,
                    'timestamp': timestamp
                })
                    
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(output_rows)
    return smooth_df_segments(df, merge_distance=5, min_duration=0.5)

def a_weighting(fs):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    a1000 = 1.9997

    nums = [(2 * np.pi * f4) ** 2 * (10 ** (a1000 / 20)), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]),
                      [1, 2 * np.pi * f2])

    return signal.bilinear(nums, dens, fs)

def moving_average(x, N=3):
    ret = pd.Series(x).rolling(N, min_periods=1).mean()
    return ret

def SileroVAD(df):
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    
    (get_speech_timestamps, _, _, _, _) = utils
    
    refined_rows = []
    target_sr = 16000 
    
    print(f"Refining {len(df)} segments...")

    for index, row in df.iterrows():
        fname = row['filename']
        date_val = row['date']
        start_sec = row['start']
        end_sec = row['end']
        timestamp = row['timestamp']
        duration = end_sec - start_sec
        
        try:
            if duration <= 0:
                continue
                
            wav_numpy, _ = librosa.load(fname, 
                                        sr=target_sr, 
                                        offset=start_sec, 
                                        duration=duration)
            
            wav = torch.from_numpy(wav_numpy)
            
            if len(wav.shape) > 1:
                wav = wav.mean(dim=0)
            
            if len(wav.shape) == 1:
                wav = wav.unsqueeze(0)

            speech_timestamps = get_speech_timestamps(
                wav, 
                model, 
                sampling_rate=target_sr,
                threshold=0.5,
                min_speech_duration_ms=250
            )
            
            if len(speech_timestamps) > 0:
                for ts in speech_timestamps:
                    rel_start_sample = ts['start']
                    rel_end_sample = ts['end']
                    
                    rel_start_sec = rel_start_sample / target_sr
                    rel_end_sec = rel_end_sample / target_sr

                    abs_start = start_sec + rel_start_sec
                    abs_end = start_sec + rel_end_sec
                    
                    refined_rows.append({
                        'filename': fname,
                        'date': date_val,
                        'start': float(abs_start), 
                        'end': float(abs_end),
                        'timestamp': timestamp
                    })
            else:
                pass

        except Exception as e:
            print(f"Error processing {fname} at {start_sec}s: {e}")

    return smooth_df_segments(pd.DataFrame(refined_rows), merge_distance=5, min_duration=0.5)

def smooth_df_segments(df_raw, merge_distance=1.0, min_duration=1.0):
    if df_raw.empty:
        return df_raw

    def clean_segments(group):
        group = group.sort_values('start')
        
        prev_end = group['end'].shift(1).fillna(group['start'].iloc[0] - (merge_distance+1))
        is_new_segment = (group['start'] - prev_end) >= merge_distance
        segment_ids = is_new_segment.cumsum()
        
        merged = group.groupby(segment_ids).agg({
            'filename': 'first', 
            'date': 'first', 
            'start': 'min', 
            'end': 'max',
            'timestamp': 'first'
        })
        
        return merged[(merged['end'] - merged['start']) >= min_duration]

    final_df = df_raw.groupby('filename', group_keys=False).apply(clean_segments).reset_index(drop=True)
    return final_df

class Diarization:
    def __init__(self, auth_token=None):
        self.token = auth_token or os.getenv("HF_TOKEN")
        
        # Check for user override
        force_cpu = os.getenv("FORCE_CPU", "false").lower() == "true"
        
        if not self.token:
            print("WARNING: No Hugging Face Token found. Diarization will fail.")
            self.pipeline = None
            return

        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.token
            )
            
            # Logic: Use GPU if available AND not forced to CPU
            if torch.cuda.is_available() and not force_cpu:
                print("🚀 Using GPU for Diarization")
                self.pipeline.to(torch.device("cuda"))
            else:
                print("🐢 Using CPU for Diarization")
                self.pipeline.to(torch.device("cpu"))
                
        except Exception as e:
            print(f"Failed to initialize Diarization pipeline: {e}")
            self.pipeline = None

    def run(self, df):
        if df.empty:
            return pd.DataFrame(columns=df.columns)

        os.makedirs("demo/temp_diarization", exist_ok=True)

        all_day_results = []
        
        for date, group in df.groupby('date'):            
            stitched_path, segment_map = self._stitch_audio(group, date)
            
            try:
                diarization = self.pipeline(stitched_path)
                
                day_rows = self._map_segments(diarization, segment_map, date)
                
                if not day_rows.empty:
                    filtered_day = self._rank_and_filter(day_rows)
                    all_day_results.append(filtered_day)
                    
            except Exception as e:
                print(f"  Error processing day {date}: {e}")
            finally:
                if os.path.exists(stitched_path):
                    os.remove(stitched_path)

        if not all_day_results:
             return pd.DataFrame(columns=['date', 'filename', 'start', 'end', 'speaker', 'timestamp'])

        return smooth_df_segments(pd.concat(all_day_results).reset_index(drop=True), merge_distance=5.0, min_duration=0.5)

    def _stitch_audio(self, group, date):
        group = group.sort_values('timestamp')
        
        full_audio = []
        segment_map = []
        
        sr = 16000
        gap = np.zeros(int(sr * 1.0), dtype=np.int16)
        
        current_sample = 0
        
        for _, row in group.iterrows():
            fname = row['filename']
            file_ts = row['timestamp']
            
            try:
                fs, data = wavfile.read(fname)
                
                if data.dtype != np.int16:
                    if np.issubdtype(data.dtype, np.floating):
                        data = (data * 32767).astype(np.int16)
                
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1).astype(np.int16)
                
                full_audio.append(data)
                
                n_samples = len(data)
                segment_map.append({
                    'global_start': current_sample,
                    'global_end': current_sample + n_samples,
                    'filename': fname,
                    'file_ts': file_ts
                })
                
                current_sample += n_samples
                
                full_audio.append(gap)
                current_sample += len(gap)
                
            except Exception as e:
                print(f"Skipping {fname} in stitching: {e}")

        stitched_data = np.concatenate(full_audio)
        temp_filename = f"demo/temp_diarization/stitched_{date}.wav"
        wavfile.write(temp_filename, sr, stitched_data)
        
        return temp_filename, segment_map

    def _map_segments(self, diarization, segment_map, date):
        results = []
        sr = 16000
        if hasattr(diarization, 'speaker_diarization'):
            diarization = diarization.speaker_diarization
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            global_start_sample = int(turn.start * sr)
            global_end_sample = int(turn.end * sr)
            
            for file_info in segment_map:
                f_start = file_info['global_start']
                f_end = file_info['global_end']
                fname = file_info['filename']
                file_ts = file_info['file_ts']
                
                overlap_start = max(global_start_sample, f_start)
                overlap_end = min(global_end_sample, f_end)
                
                if overlap_start < overlap_end:
                    local_start_sec = (overlap_start - f_start) / sr
                    local_end_sec = (overlap_end - f_start) / sr
                    
                    seg_ts = file_ts + pd.to_timedelta(local_start_sec, unit='s')
                    
                    results.append({
                        'date': date,
                        'filename': fname,
                        'start': local_start_sec,
                        'end': local_end_sec,
                        'speaker': speaker,
                        'timestamp': seg_ts
                    })
        
        return pd.DataFrame(results)

    def _rank_and_filter(self, df, alpha=0.7, beta=0.3):
        stats = []
        speakers = df['speaker'].unique()
        
        for spk in speakers:
            spk_data = df[df['speaker'] == spk]
            duration = (spk_data['end'] - spk_data['start']).sum()
            hours = spk_data['timestamp'].dt.hour.nunique()
            stats.append({'speaker': spk, 'duration': duration, 'coverage': hours})
        
        if not stats:
            return pd.DataFrame(columns=df.columns)
            
        stats_df = pd.DataFrame(stats)
        
        d_min, d_max = stats_df['duration'].min(), stats_df['duration'].max()
        stats_df['norm_duration'] = 1.0 if d_max == d_min else (stats_df['duration'] - d_min) / (d_max - d_min)
        
        c_min, c_max = stats_df['coverage'].min(), stats_df['coverage'].max()
        stats_df['norm_coverage'] = 1.0 if c_max == c_min else (stats_df['coverage'] - c_min) / (c_max - c_min)
        
        stats_df['score'] = (alpha * stats_df['norm_duration']) + (beta * stats_df['norm_coverage'])
        
        primary = stats_df.sort_values(by='score', ascending=False).iloc[0]['speaker']
        print(f"  Primary Speaker identified: {primary} (Score: {stats_df['score'].max():.2f})")
        
        return df[df['speaker'] == primary].copy()
