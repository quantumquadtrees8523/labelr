from dspy.predict import aggregation
from utils.gemini_agent import GeminiAgent

class ActiveLearner():
    """This is the active learning agent. We aim to use this as a machine-human interface
    that will allow us to properly label the datapoints that the machine is least confident
    about.
    """

    def __init__(self, num_learning_iterations, task, init_training_data=[], unlabeled_data=[]):
        """All incoming inputs must be pre-processed and cleaned.
        """
        self.num_learning_iterations = num_learning_iterations
        self.task = task

        self.init_training_data = init_training_data
        self.unlabeled_data = unlabeled_data
        self.batch_size = len(unlabeled_data) // num_learning_iterations

        self.agents = [GeminiAgent() for i in range(5)]
    
    def predict_label(self, input):
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
        self.fine_tune_model(self.init_training_data)
        for iteration in range(self.num_learning_iterations):
            print("=====")
            print("=====")
            print("=====")
            print("=====")
            print("=====")
            print(self.predict_label(self.task))
            print("=====")
            print("=====")
            print("=====")
            print("=====")
            print("=====")
            # pre_label_batch = self.unlabeled_data[i * self.batch_len:(i + 1) * self.batch_len]
            # post_label_batch = [self.predict_label(inp) for inp in self.pre_label_batch]
            # human_refined_labels = self.gather_human_feedback(post_label_batch)
            # self.fine_tune_model(human_refined_labels)
        return 
    
    def create_final_dataset(self):
        return [self.predict_label(inp) for inp in self.unlabeled_data] + self.training_data_batch