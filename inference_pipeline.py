from typing import List
import uuid
import pandas as pd
import dspy
from dspy import aggregation, Signature, Prediction
from dspy.signatures import make_signature
from pydantic import Extra
from data_types.post_inference_set import PostInferenceSet
from data_types.pre_inference_set import PreInferenceSet
from language_model_utils.gemini_client import GeminiClient
import traceback

# class QuerySignature(dspy.Signature):
#         def __init__(self, csv_column_names, task, output_name, output_description):
#             inputs_dict: dict = dspy.Signature.
#             inputs_dict['task'] = task
#             for column_name in csv_column_names:
#                 inputs_dict[column_name] = dspy.InputField()
#             inputs_dict[output_name] = dspy.OutputField()
#             self.__class__.__doc__ = output_description
#             s = dspy.Signature(inputs_dict)

class InferencePipeline:

    def __init__(self, pre_inference_set: PreInferenceSet, task="classification", output_filename="DEFAULT", output_description="label description", num_agents=1) -> None:
        self.pre_inference_set = pre_inference_set
        columns: List = self.pre_inference_set.get_columns()
        instructions_dict: dict = dict()
        instructions_dict['task'] = dspy.InputField()
        for column_name in columns:
            instructions_dict[column_name] = dspy.InputField()
        instructions_dict['llm_inference'] = dspy.OutputField(desc=output_description)
        self.output_filename = output_filename
        signature: type[Signature] = make_signature(instructions_dict)
        self.agents = [GeminiClient(signature) for i in range(num_agents)]
        self.post_inference_set = PostInferenceSet()
        self.feature_id = uuid.uuid4()

    def run(self) -> bool:
        for index, record in enumerate(self.pre_inference_set):
            print("predicting record #: ", index)
            try:
                pred: Prediction = self.inference(record)
                # This `str` of the record name is duct tape. how do you actually want to store it?
                self.post_inference_set.add_record(str(record.name), self.feature_id, pred)
            except Exception:
                print(traceback.print_exc())

        return self.commit()

    def inference(self, record: pd.Series) -> Prediction:
        # Get consensus and return it.
        return aggregation.majority([agent.forward(record.to_dict()) for agent in self.agents])
    
    def commit(self) -> bool:
        post_inference_df: pd.DataFrame = self.post_inference_set.get_write_format()
        post_inference_df.to_csv(self.pre_inference_set.input_filename[:-4] + "_" + self.output_filename + "_RESULT.csv")
        return True
