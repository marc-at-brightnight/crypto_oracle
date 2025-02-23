import polars as pl


def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
    """
    Handles missing values, ensures consistency in data types. Return preprocessed data as dataframe.
    """
    return data.select(pl.all().forward_fill().backward_fill())
