import pandas as pd

def load_and_format_dataset(file_path):
    df = pd.read_csv(file_path)
    formatted_data = df.apply(lambda row: f"### Human: {row['query']} ### Assistant: {row['response']}", axis=1)
    return formatted_data
