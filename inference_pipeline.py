import dspy
from  data_types.pre_inference_set import InferenceSet
from language_model_utils.gemini_client import GeminiClient

class InferencePipeline:
    def __init__(self, inference_set: InferenceSet, signature: dspy.Signature) -> None:
        self.inference_set = inference_set
        # self.agents = [GeminiClient(signature) for i in range(1)]

    def run(self):


