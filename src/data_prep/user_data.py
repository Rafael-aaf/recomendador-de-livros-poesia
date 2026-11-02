import pandas as pd


def load_user_csv(path):
    df = pd.read_csv(path, dtype={"Book Id": str, "ISBN13": str})
    df["rating"] = pd.to_numeric(df["My Rating"], errors="coerce")
    df["rating"] = df["rating"].where(df["rating"] > 0)
    return df