import pandas as pd
import uuid 

class PreInferenceSet():
    # Function to get specific columns by index.
    def get_column_name(self, index: int) -> str:
        return self.dataset_df.columns[index]
    
    def __init__(self, filename: str):
        self.dataset_df: pd.DataFrame = pd.read_csv(filename)
        self.dataset_df['record_id'] = [uuid.uuid4() for i in range(self.dataset_df.shape[0])]
        # Unsure if we will need this logic in the future.
        #
        # training_data: list = []
        # for features in self.dataset_df.values.tolist():
        #     example: dict = {}
        #     # Create an example that will be fed into DSPy Dataset class.
        #     # For example: {'label': pair[0], 'text': pair[1]}
        #     for i in range(len(features)):
        #         example.update({self.get_column_name(i): features[i]})
        #     training_data.append(example)

    def get_row(self, index) -> pd.DataFrame:
        return self.dataset_df.iloc[index]
    
    def __iter__(self):
        self.index: int = 0  # Reset the index for a new iteration
        return self

    def __next__(self) -> pd.Series:
        if self.index < len(self.dataset_df):
            row = self.dataset_df.iloc[self.index]
            self.index += 1
            return row
        else:
            raise StopIteration
    

        
