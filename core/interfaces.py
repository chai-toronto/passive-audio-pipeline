from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, List, Dict

class BaseFilter(ABC):
    """Abstract base class for all data filters."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe, applies logic, returns filtered dataframe."""
        pass

class BaseFeatureExtractor(ABC):
    """Defines how features are extracted from raw data."""
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe where columns are features and rows are samples."""
        pass

class BaseTransformation(ABC):
    """Defines the 'reverse transformation' (e.g., Denoising)."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the transformation (e.g., noise reduction) to the data."""
        pass

class BaseCharacteristicMeasure(ABC):
    """Measures the strength of a characteristic (e.g., Noise Level)."""
    @abstractmethod
    def measure(self, df: pd.DataFrame) -> float:
        """Returns a scalar representing the magnitude of the characteristic."""
        pass

class BaseDistanceMeasure(ABC):
    """Measures the shift in features between two states."""
    @abstractmethod
    def calculate(self, features_a: pd.DataFrame, features_b: pd.DataFrame) -> pd.Series:
        """Returns a Series of distances/rankings, one per feature."""
        pass