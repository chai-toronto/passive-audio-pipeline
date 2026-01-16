from typing import List
import pandas as pd
from core.interfaces import BaseFilter

class FilterPipeline:
    def __init__(self):
        self._steps: List[BaseFilter] = []

    def add_filter(self, filter_step: BaseFilter) -> 'FilterPipeline':
        self._steps.append(filter_step)
        return self

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flows the data through all registered filters sequentially."""
        current_df = df.copy()
        for step in self._steps:
            current_df = step.apply(current_df)
        return current_df