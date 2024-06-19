import uuid
import pandas as pd
from dspy import aggregation, Signature, Prediction
from data_types.post_inference_set import PostInferenceSet
from data_types.pre_inference_set import PreInferenceSet
from language_model_utils.gemini_client import GeminiClient

class InferencePipeline:
    def __init__(self, pre_inference_set: PreInferenceSet, signature: Signature, num_agents=1) -> None:
        self.pre_inference_set = pre_inference_set
        self.agents = [GeminiClient(signature) for i in range(num_agents)]
        self.post_inference_set = PostInferenceSet()
        self.feature_id = uuid.uuid4()

    def run(self) -> bool:
        for record in self.pre_inference_set[:1]:
            try:
                pred: Prediction = self.inference(record)
                self.post_inference_set.add_record(record.record_id, self.feature_id, pred)
            except:
                print("error")
        return True

    def inference(self, record: pd.Series) -> Prediction:
        # Get consensus and return it.
        return aggregation.majority([agent.forward(record.to_dict()) for agent in self.agents])
    
    def commit(self) -> bool:
        post_inference_df: pd.DataFrame = self.post_inference_set.get_write_format()
        post_inference_df.to_csv(self.pre_inference_set.input_filename[:-5] + "_RESULT.csv")
        return True
