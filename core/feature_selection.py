import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from core.interfaces import (
    BaseFeatureExtractor, 
    BaseTransformation, 
    BaseCharacteristicMeasure, 
    BaseDistanceMeasure
)

@dataclass
class SelectionResult:
    robust_features: List[str]
    sensitive_features: List[str]
    characteristic_delta: float
    feature_distances: pd.Series

class RobustnessSelector:
    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        transformation: BaseTransformation,
        characteristic_measure: BaseCharacteristicMeasure,
        distance_measure: BaseDistanceMeasure,
        change_threshold: float = 0.1  # Configurable threshold for 'changing' vs 'same'
    ):
        self.extractor = extractor
        self.transformation = transformation
        self.char_measure = characteristic_measure
        self.dist_measure = distance_measure
        self.threshold = change_threshold

    def select(self, df: pd.DataFrame, noise_thresh=2.5, correlation_threshold=0.75) -> Dict[str, SelectionResult]:
        
        raw_char_score = self.char_measure.measure(df)    
        raw_features = self.extractor.extract(df)

        transformed_df = self.transformation.apply(df)

        trans_char_score = self.char_measure.measure(transformed_df)
        trans_features = self.extractor.extract(transformed_df)

        scores = pd.DataFrame({
            'moslqo_pre': raw_char_score['moslqo'],
            'moslqo_post': trans_char_score['moslqo']
        })

        valid_mask = scores['moslqo_post'] > noise_thresh
        
    
        idx_was_noisy = scores[valid_mask & (scores['moslqo_pre'] < noise_thresh)].index
        idx_was_clean = scores[valid_mask & (scores['moslqo_pre'] > noise_thresh)].index

        corr_control = self.dist_measure.calculate(
            raw_features.loc[idx_was_clean], 
            trans_features.loc[idx_was_clean]
        ).fillna(1)

        corr_enhanced = self.dist_measure.calculate(
            raw_features.loc[idx_was_noisy], 
            trans_features.loc[idx_was_noisy]
        ).fillna(1)

        analysis = pd.DataFrame({
            'corr_control': corr_control,
            'corr_enhanced': corr_enhanced
        })

        robust_feats = analysis[
            (analysis['corr_control'].abs() >= correlation_threshold) & 
            (analysis['corr_enhanced'].abs() >= correlation_threshold)
        ].index.tolist()

        enhanced_feats = analysis[
            (analysis['corr_control'].abs() >= correlation_threshold) & 
            (analysis['corr_enhanced'].abs() < correlation_threshold)
        ].index.tolist()

        return {
            'robust': robust_feats,
            'enhanced': enhanced_feats,
        }