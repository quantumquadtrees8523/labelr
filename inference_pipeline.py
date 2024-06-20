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
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class InferencePipeline:

    def __init__(self, pre_inference_set: PreInferenceSet, task="classification", output_filename="DEFAULT", output_description="label description", num_agents=1) -> None:
        print("task: " + task)
        print("output filename: " + output_filename)
        print("output: " + output_description)
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
        self.commit(self.pre_inference_set.dataset_df, output_filename + "SOURCE")

    def run(self) -> bool:
        print("predicting ", str(len(self.pre_inference_set)) + " records.")
        print("-------")
        for index, record in enumerate(self.pre_inference_set):
            try:
                pred: Prediction = self.inference(record)
                # This `str` of the record name is duct tape. how do you actually want to store it?
                self.post_inference_set.add_record(str(record.name), self.feature_id, pred)
            except:
                traceback.print_exc()

        return self.commit(self.post_inference_set.dataset_df, self.output_filename + "RESULT")

    def inference(self, record: pd.Series) -> Prediction:
        # Get consensus and return it.
        return aggregation.majority([agent.forward(record.to_dict()) for agent in self.agents])
    
    def commit(self, df: pd.DataFrame, index_name: str) -> bool:
        es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        # indices = es.indices.get_alias(index=self.output_filename + index_name)
        # This is bad logic. It's overly general and you are only deleting the one index
        # Modify this.
        # for index in indices:
        #     # Delete each non-system index
        #     es.indices.delete(index=index)
        #     print(f"Deleted index: {index}")
        actions = [
            {
                "_index": index_name,
                "_source": row.to_dict()
            }
            for _, row in df.iterrows()
        ]
        # Use the bulk helper function to index data
        bulk(es, actions)
        # try:
        #     bulk(es, actions)
        # except:
        #     print("ERROR COMMITTING TO ELASTICSEARCH.")
        return True
