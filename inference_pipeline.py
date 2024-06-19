import pandas as pd
from dspy import aggregation, Signature, Prediction
from data_types.post_inference_set import PostInferenceSet
from data_types.pre_inference_set import PreInferenceSet
from language_model_utils.gemini_client import GeminiClient

class InferencePipeline:
    def __init__(self, pre_inference_set: PreInferenceSet, signature: Signature) -> None:
        self.pre_inference_set = pre_inference_set
        self.agents = [GeminiClient(signature) for i in range(1)]
        self.post_inference_set = PostInferenceSet()

    def run(self) -> bool:
        for record in self.pre_inference_set:
            try:
                pred: Prediction = self.inference(record)
                
            except:
                print("error")
        return True

    def inference(self, record: pd.Series) -> Prediction:
        # Get consensus and return it.
        return aggregation.majority([agent.forward(record.to_dict()) for agent in self.agents])
