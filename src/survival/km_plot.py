import os
import pandas as pd

def load_km_data_from_csv(folder_path, cancer_type, subtype):
    filename = f"{cancer_type}_{subtype}_gpt_risk_predictions_-1.csv"
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    labels = df["predicted_label"].values
    times = df["event_time"].values
    events = df["event_indicator"].astype(bool).values
    return labels, times, events