import pandas as pd
import datasets
from dspy.datasets.dataset import Dataset

class MyDataset(Dataset):
    # Function to get specific columns by index
    def get_columns_by_index(self, df, record, indices):
        return {list(df.columns)[i]: record[i] for i in indices}
    
    def __init__(self, input_file, training_set_size=50):
        super().__init__()
        self.dataset = pd.read_csv(input_file)
        label_to_text_pairing = self.dataset.values.tolist()
        self.training_set_size = training_set_size
        training_data = []
        csv_write_dataframe = pd.DataFrame(columns=['label', 'text'])
        for pair in label_to_text_pairing:
            example = {'label': pair[0], 'text': pair[1]}
            pd.concat([csv_write_dataframe, pd.DataFrame([example])], ignore_index=True)
            training_data.append(example)
        # Store in memory as a dspy Dataset object
        self._train = training_data[:]