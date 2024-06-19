import pandas as pd
from dspy.predict import aggregation
from language_model_utils.gemini_client import GeminiClient


class InferenceAgent():
    """This is the active learning agent. We aim to use this as a machine-human interface
    that will allow us to properly label the datapoints that the machine is least confident
    about.
    """

    def __init__(self, write_filename, inference_data):
        """All incoming inputs must be pre-processed and cleaned.
        """
        self.write_filename = write_filename
        self.inference_data = inference_data
        self.agents = [GeminiClient() for i in range(1)]

    
    def predict_label_by_majority_vote(self, input):
        """Must also return a confidence score.
        """
        # Get consensus and return it.
        return aggregation.majority([agent.forward(input) for agent in self.agents])


    def gather_human_feedback(self, data_lst=[]):
        """This will go through the list and find all elements with a low threshold score
        and get human input on their values. It will then update the label with the human
        input.
        """
        return
    
    def fine_tune_model(self, training_data_batch):
        """Fine-tune the llm.
        """
        return
    
    def learn(self):
        # self.fine_tune_model(self.init_training_data)
        results = pd.DataFrame(columns=['label', 'text'])
        print("Predicting labels for: " + str(len(self.inference_data)) + " records.")
        print(" ")
        for i in range(len(self.inference_data)):
            print("record #" + str(i + 1))
            datapoint = self.inference_data[i]
            text = datapoint['text']
            # pred = self.predict_label_by_majority_vote(text)
            # category = pred['category']
            # results = pd.concat([results, pd.DataFrame([{'label': category, 'text': text}])], ignore_index=True)
        results.to_csv(self.write_filename, index=False)
        return 
    
    def create_final_dataset(self):
        return [self.predict_label_by_majority_vote(inp) for inp in self.inference_data]
