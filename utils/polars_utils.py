import polars as pl

class PolarsUtils:
    def print_df(df: pl.DataFrame):
        df_string = ""
        df_string+=(" ".join(df.columns))
        for row in df.rows():
            df_string+=(" ".join(map(str, row)))
        return df_string