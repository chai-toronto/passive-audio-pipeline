import pandas as pd
import numpy as np
import os

from core.filter import FilterPipeline
from core.feature_selection import RobustnessSelector
from demo.noise_pipeline_demo import NoiseReduce 
from demo.noise_pipeline_demo import OpenSMILE
from demo.noise_pipeline_demo import VisQOL
from demo.noise_pipeline_demo import PearsonCoefficient
from demo.noise_pipeline_demo import SilenceFilter, VAD, PrimarySpeaker

def main():
    data = pd.DataFrame(os.listdir("demo/audio_samples"))
    data['date'] = pd.Timestamp("2026-01-01")
    offsets = np.linspace(0, 24, len(data), endpoint=False)
    data['timestamp'] = data['date'] + pd.to_timedelta(offsets, unit='h')
    data = data.rename({0: "filename"}, axis=1)
    data['filename'] = "demo/audio_samples/" + data['filename']

    filter = FilterPipeline()
    filter.add_filter(SilenceFilter())
    filter.add_filter(VAD())
    filter.add_filter(PrimarySpeaker())

    clean_data = filter.process(data)

    selector = RobustnessSelector(
        extractor=OpenSMILE(),
        transformation=NoiseReduce(),
        characteristic_measure=VisQOL(),
        distance_measure=PearsonCoefficient(),
        change_threshold=0.05
    )

    results = selector.select(clean_data)

    print(f"Robust Features: {results['robust']}")
    print(f"Enhanced Features: {results['enhanced']}")

if __name__ == "__main__":
    main()