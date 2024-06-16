class ActiveLearner():
    """This is the active learning agent. We aim to use this as a machine-human interface
    that will allow us to properly label the datapoints that the machine is least confident
    about.
    """

    def __init__(self, init_training_data=[], unlabeled_data=[], num_learning_iterations):
        """All incoming inputs must be pre-processed and cleaned.
        """
        self.init_training_data = init_training_data
        self.unlabeled_data = unlabeled_data
        self.num_learning_iterations = num_learning_iterations
        self.batch_len = 
    
    def predict_label(self, input):
        """Must also return a confidence score.
        """
        return 

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
