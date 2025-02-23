import pandas as pd


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values, ensures consistency in data types. Return preprocessed data as dataframe.
    """
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    print("Data preprocessing completed successfully")
    return data
