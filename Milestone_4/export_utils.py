import pandas as pd

def export_csv(df, filename="output.csv"):
    return df.to_csv(index=False).encode('utf-8')