import polars as pl

class PolarsUtils:
    def print_df(df: pl.DataFrame):
        string_df = ""
        string_df+=(" ".join(df.columns))
        for row in df.rows():
            string_df+=(" ".join(map(str, row)))+"\n"
        return string_df