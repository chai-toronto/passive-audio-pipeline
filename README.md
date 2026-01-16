# Audio Feature Robustness Pipeline

## Project Description

This repository implements a modular framework for evaluating and selecting features based on their robustness to noise and their responsiveness to enhancement algorithms on your data. 

Initially designed for noise reduction in audio analysis, this pipeline separates the feature engineering process into discrete stages: **Filtering** (VAD, Diarization), **Transformation** (Denoising), and **Selection** (Feature Stability).

The core objective is to identify two distinct classes of features:

1. **Robust Features:** Features that remain statistically stable regardless of degredation artifacts (in our case noise within audio).
2. **Enhanced Features:** Features that become enhanced when the signal is transformed (in our case when audio is denoised).

This implementation includes a demonstration using **OpenSMILE** for feature extraction, **Silero VAD** and **Pyannote** for filtering, **VisQOL** for quality assessment, and **Noise Reduce** for transformation of the signal.

## Prerequisites

* **Python 3.10+**
* **uv**
* **System Dependencies:** `libsndfile` and `ffmpeg` are required for audio processing.
* Ubuntu/Debian: `sudo apt-get install libsndfile1 ffmpeg`
* macOS: `brew install libsndfile ffmpeg`



### A Note on VisQOL

This pipeline relies on **Google VisQOL** (Virtual Speech Quality Objective Listener) for the characteristic measurement step. VisQOL does not have a standard PyPI package and requires building the Python bindings from source using Bazel.

1. Clone the [VisQOL repository](https://github.com/google/visqol).
2. Follow their instructions to build the Python API (`bazel build :visqol_lib_py`).
3. Ensure the generated `visqol` protobufs and library files are accessible in your `PYTHONPATH` or installed in your environment.

## Installation

This project uses `uv` for fast dependency management.

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/audio-robustness-pipeline.git
cd audio-robustness-pipeline

```


2. **Create a virtual environment:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```


3. **Install dependencies:**
```bash
uv pip install -r requirements.txt

```


4. **Hugging Face Authentication (Pyannote):**
The speaker diarization module uses `pyannote.audio`, which requires an access token.
1. Accept `pyannote/speaker-diarization-3.1` user conditions on Hugging Face.
2. Set your token in the environment or update `demo/noise_pipeline_demo.py`:


```python
# In demo/noise_pipeline_demo.py
self.pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="YOUR_HUGGING_FACE_TOKEN"
)

```



## Directory Overview

```text
.
├── core/                     # Abstract Framework Logic
│   ├── feature_selection.py  # Logic for comparing feature sets (RobustnessSelector)
│   ├── filter.py             # Pipeline logic for chaining data filters
│   └── interfaces.py         # Abstract Base Classes (ABC) for extensibility
├── demo/                     # Concrete Implementations & Demo Assets
│   ├── audio.py              # Wrappers for VAD (Silero) and Diarization (Pyannote)
│   ├── audio_samples/        # Raw input audio for the demo
│   ├── noise_pipeline_demo.py# Concrete classes implementing core interfaces
│   └── reference_audio/      # Clean reference files required by VisQOL
├── main.py                   # Entry point for the demonstration
└── requirements.txt          # Python dependencies

```

## Running the Demo

The included `main.py` runs a full pass of the pipeline on the dummy audio provided in `demo/audio_samples`.

1. Ensure you have reference audio files (clean speech samples) in `demo/reference_audio` if using VisQOL in audio mode.
2. Run the pipeline:

```bash
python main.py

```

### Expected Output

The script will process the audio, apply filters, and output the lists of features identified as robust or enhanced:

```text
Refining 16 segments...
Primary Speaker identified: SPEAKER_01 (Score: 0.85)
...
Robust Features: ['F0final_sma_stddev', 'loudness_sma3_amean', ...]
Enhanced Features: ['pcm_zcr_sma_stddev', 'jitterLocal_sma_amean', ...]

```

## Customizing the Pipeline

The framework is designed to be agnostic to the specific libraries used. To use your own feature extractors, filters, or distance measures, you must implement the interfaces defined in `core/interfaces.py`.

### 1. Implementing a Custom Feature Extractor

If you wish to use `librosa` or `wav2vec` instead of OpenSMILE, create a class that inherits from `BaseFeatureExtractor`.

```python
from core.interfaces import BaseFeatureExtractor
import librosa
import numpy as np
import pandas as pd

class LibrosaMFCC(BaseFeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        features = []
        for _, row in df.iterrows():
            y, sr = librosa.load(row['filename'], sr=16000)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            # Create a dictionary of features
            feat_dict = {f'mfcc_{i}': val for i, val in enumerate(mfcc)}
            features.append(feat_dict)
        return pd.DataFrame(features)

```

### 2. Implementing a Custom Filter

To add a specific bandpass filter or a different VAD:

```python
from core.interfaces import BaseFilter

class MyBandpassFilter(BaseFilter):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Logic to process audio files referenced in df
        # Return updated DataFrame (potentially with modified filenames)
        return df

```

### 3. Integrating into Main

Swap your new classes into the pipeline in `main.py`:

```python
# ... inside main()
selector = FeatureSelector(
    extractor=LibrosaMFCC(),       # Your custom extractor
    transformation=NoiseReduce(),
    characteristic_measure=VisQOL(),
    distance_measure=PearsonCoefficient(),
    change_threshold=0.05
)

```

## Citation

If you use this repository in your research, please cite:

[Insert Citation Here]

## License

[Insert License Here]
