import pandas as pd
import datasets
import config

file_path = "./phiDataset.csv"

def load_local_dataset(file_path):
    print("Loading dataset from - ", file_path, "...")
    dataset_df = pd.read_csv(file_path)
    return dataset_df

def prepare_prompts_responses(dataset_df):
    print("Preparing Prompt and Assistant....")
    prompts_responses = []
    for _, record in dataset_df.iterrows():
        prompt_text = record['Hook']
        response_text = record['Body ']
        prompt_response = "### Human: " + prompt_text + " ### Assistant: " + response_text
        prompts_responses.append({config.DATASET_TEXT_FIELD: prompt_response})
    
    return pd.DataFrame(prompts_responses)

def preparedata(mode):
    print("Preparing data for - ", mode, "...")
    dataset_df = load_local_dataset(config.DATASET_PATH)
    prompts_responses_df = prepare_prompts_responses(dataset_df)
    prompts_responses_dataset = datasets.Dataset.from_pandas(prompts_responses_df)
    return prompts_responses_dataset
