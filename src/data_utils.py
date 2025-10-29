import pandas as pd
import os

def load_data(path="data/corona.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Basic cleaning example
    df = df.drop_duplicates()
    # Try to parse dates if present
    for c in ['ObservationDate', 'Date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df
