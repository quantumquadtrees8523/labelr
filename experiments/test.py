from inference_agent import InferenceAgent
from data_types.my_dataset import MyDataset

"""
Experiment 1 Details:
---------------------
The input is product descriptions and the label is one of four:
    - Household
    - Clothing & Accessories
    - Electronics
    - Books

100 records.
4 classes.
  1. 25.73 seconds for 5 predict_label calls.
  2. 21.76 seconds for 5 predict_label calls.
  3. 3:38.92 minutes for 48 predict_label calls.
Pretty much 100%
"""
# This is the core file the test is based on. The results that will be created
# are predictions of the text field in the below file.
# dataset = MyDataset('datasets/data/ecommerceDataset.csv')

# NUM_LEARNING_ITERATIONS = 1
# print(len(dataset.train))
# learner = InferenceAgent('datasets/data/ecommerceDataset_results.csv', NUM_LEARNING_ITERATIONS, dataset.train)
# learner.learn()