import pandas as pd
from dspy import TypedChainOfThought
from typing import List

class PostInferenceSet:
    def __init__(self) -> None:
        columns = ["record_id", "feature_id", "llm_inference", "llm_rationale"]
        self.dataset_df: pd.DataFrame = pd.DataFrame(columns=columns)
    
    def add_record(self, record_id: str, feature_id: str, prediction):
        # For this to work the signature of the dspy agent must return `label`.`
        post_inference_record: dict = {'record_id': record_id, 'feature_id': feature_id, 'llm_inference': prediction.label, 'llm_rationale': prediction.rationale}
        # self.dataset_df.append(post_inference_record)
