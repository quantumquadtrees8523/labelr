import pandas as pd
import uuid 

class PreInferenceSet():
    def __init__(self, filename: str):
        self.input_filename = filename
        if self.input_filename[-4:] == '.tsv':
            self.dataset_df: pd.DataFrame = pd.read_csv(filename, sep="\t")
        else:
            self.dataset_df: pd.DataFrame = pd.read_csv(filename)
        self.dataset_df['record_id'] = [uuid.uuid4() for i in range(self.dataset_df.shape[0])]
        # Create a CSV with the record_id included.
        self.dataset_df.to_csv(self.input_filename)

    # Function to get specific columns by index.
    def get_column_name(self, index: int) -> str:
        return self.dataset_df.columns[index]
    
    def get_row(self, index) -> pd.DataFrame:
        return self.dataset_df.iloc[index]
    
    def __iter__(self):
        self.index: int = 0  # Reset the index for a new iteration
        return self

    def __next__(self) -> pd.Series:
        if self.index < 100: # len(self.dataset_df):
            row = self.dataset_df.iloc[self.index]
            self.index += 1
            return row
        else:
            raise StopIteration
        
    # def __getitem__(self, index) -> pd.Series:
    #     return self.dataset_df.iloc[index]
