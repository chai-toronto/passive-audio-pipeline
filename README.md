## 1. Project Description

This repository implements a modular framework for selecting features based on their robustness to noise and their responsiveness to enhancement algorithms.

Initially designed for noise reduction in audio analysis, this pipeline separates the feature engineering process into discrete stages: **Filtering**, **Transformation**, and **Selection**. The initial design relied on the following tools for the three stages:
* **Filtering:** Silence Removal, Voice Activity Detection (Silero), Diarization (Pyannote)
* **Transformation:** Denoising (Noise Reduce)
* **Selection:** Correlation & Quality Assessment (VisQOL)

The core objective is to identify two distinct classes of features:

1.  **Robust Features:** Features that remain stable regardless of degradation artifacts (in our case, noise within audio).
2.  **Enhanced Features:** Features that become significantly improved when the signal is transformed (in our case, when audio is denoised).

This implementation includes a demonstration using **OpenSMILE** for feature extraction, **Silero VAD** and **Pyannote** for filtering, **VisQOL** for quality assessment, and **Noise Reduce** for signal transformation.

While main.py executes a demonstration specifically focused on noise removal in speech, the framework is architected for broad extensibility. It is designed to be adapted for other domains and signal types—such as physiological data or time-series sensor readings—allowing researchers to swap in alternative tools for feature extraction, filtering, or transformation to test feature robustness in entirely new contexts.

---

## 2. Installation & Execution

**Prerequisites:**
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* A [Hugging Face Token](https://huggingface.co/settings/tokens) (Required for Pyannote).
    * *Note: You must accept the user conditions for `pyannote/speaker-diarization-3.1` on their model page.*

**Steps:**

1.  **Build the Image** (approx. 5-10 mins):
    ```bash
    docker build --platform linux/amd64 -t audio-pipeline-artifact .
    ```

2.  **Configure Environment:**
    Create a file named `.env` in the root directory. You may refer to the .env.example file for reference:
    ```ini
    HF_TOKEN=hf_YourTokenHere
    NOISE_THRESHOLD=2.5
    CORRELATION_THRESHOLD=0.75
    ```

3.  **Run the Pipeline:**
    ```bash
    docker run --rm -it --env-file .env audio-pipeline-artifact
    ```

4.  **VisQOL Validation**
    To verify that the complex dependencies (VisQOL and its protobuf bindings) are installed correctly without waiting for the full pipeline, users can run this sanity check:
    ```bash
    docker run --rm audio-pipeline-artifact python -c "from visqol import visqol_lib_py; print('SUCCESS: VisQOL is installed correctly.')"
    ```

## 3. Configuration Parameters
The pipeline behavior can be tuned using environment variables (in `.env`).

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *Required* | Hugging Face authentication token for Speaker Diarization. |
| `NOISE_THRESHOLD` | `2.5` | The minimum VisQOL score improvement required to classify a file as clean/noisy. |
| `CORRELATION_THRESHOLD` | `0.75` | The minimum Pearson correlation coefficient required to classify a feature as stable/changed. |
| `CHANGE_THRESHOLD` | `0.05` | The sensitivity for detecting signal changes during the filtering stage. |

---

## 4. Directory Overview

```text
.
├── core/                     # Abstract Framework Logic
│   ├── feature_selection.py  # Logic for comparing feature sets (RobustnessSelector)
│   ├── filter.py             # Pipeline logic for chaining data filters
│   └── interfaces.py         # Abstract Base Classes (ABC) for extensibility
├── demo/                     # Concrete Implementations & Demo Assets
│   ├── audio.py              # Wrappers for VAD (Silero) and Diarization (Pyannote)
│   ├── audio_samples/        # Raw input audio for the demo
│   ├── noise_pipeline_demo.py# Concrete classes implementing core interfaces for the purposes of the demo
│   └── reference_audio/      # Clean reference files required by VisQOL
├── main.py                   # Entry point for the demonstration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Automated build environment
└── .env                      # User configuration (Tokens & Thresholds)
```

## 5. Understanding the Output from running main.py
The specific features identified will vary based on your input data, chosen tools, and threshold settings.

For the sample audio included in this artifact, running the pipeline with the **strict thresholds defined in our paper** yields a populated set of **Robust Features** but an empty set of **Enhanced Features**. This is expected behavior for this specific sample subset and demonstrates the pipeline correctly filtering out candidates that do not meet the quality improvement metrics required by the configuration.

## 6. Extending the Implementation

The framework is built on a set of abstract base classes defined in `core/interfaces.py`. This allows you to swap out any component—whether it is the feature extractor, the noise filter, or the quality measure—without rewriting the core logic.

### 7. Implementing a Custom Feature Extractor
To use a different feature set (e.g., `librosa` MFCCs instead of OpenSMILE), create a class that inherits from `BaseFeatureExtractor`.

```python
from core.interfaces import BaseFeatureExtractor
import librosa
import numpy as np
import pandas as pd

class LibrosaMFCC(BaseFeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        features = []
        # The input DataFrame contains 'filename', 'start', and 'end' columns
        for _, row in df.iterrows():
            # Load the specific segment
            y, sr = librosa.load(row['filename'], sr=16000)

            # Extract features (e.g., MFCCs averaged over time)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

            # Return a dict where keys are feature names
            features.append({f'mfcc_{i}': val for i, val in enumerate(mfcc)})

        return pd.DataFrame(features)
```

New classes, once defined and implemented can be added into the pipeline, as done in the demo with the respective classes and objects. 

## 7. Reference

For full details on the methodology, background, and results, please refer to our paper:

**Addressing Extra Voices and Background Noise in Continuous Speech Monitoring: A Case Study on COPD**
2026 IEEE International Conference on Pervasive Computing and Communications (PerCom)

```bibtex
@inproceedings{liaqat2026addressing,
  title={Addressing Extra Voices and Background Noise in Continuous Speech Monitoring: A Case Study on COPD},
  author={Liaqat, Salaar and Liaqat, Daniyal and Son, Tatiana and Wu, Robert and Gershon, Andrea and de Lara, Eyal and Mariakakis, Alex},
  booktitle={2026 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  year={2026}
}
