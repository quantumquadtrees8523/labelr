from uuid import UUID
import pandas as pd
from dspy import TypedChainOfThought
from typing import List

class PostInferenceSet:
    def __init__(self) -> None:
        columns = ["record_id", "feature_id", "llm_inference", "llm_rationale"]
        self.dataset_df: pd.DataFrame = pd.DataFrame(columns=columns)
        self.dataset_df.set_index('record_id', inplace=True)
    
    def add_record(self, record_id: str, feature_id: UUID, prediction):
        # For this to work the signature of the dspy agent must return `label`.`
        post_inference_record: dict = {'record_id': record_id, 'feature_id': feature_id, 'llm_inference': prediction.llm_inference, 'llm_rationale': prediction.rationale}
        record_df: pd.DataFrame = pd.DataFrame([post_inference_record])
        record_df.set_index('record_id', inplace=True)
        self.dataset_df: pd.DataFrame = pd.concat([self.dataset_df, record_df])

    def get_write_format(self) -> pd.DataFrame:
        return self.dataset_df