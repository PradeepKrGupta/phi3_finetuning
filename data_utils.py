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
    user_prompters = dataset_df[(dataset_df.role == "prompter")]
    user_prompters = user_prompters.set_index("message_id")
    assistants = dataset_df[(dataset_df.role == "assistant") & (dataset_df["rank"] == 0.0)]
    
    prompts_responses = []
    for _, record in assistants.iterrows():
        prompt_text = user_prompters.loc[record.parent_id, 'text']
        prompt_response = "### Human: " + prompt_text + " ### Assistant: " + record['text']
        prompts_responses.append(prompt_response)
    assistants[config.DATASET_TEXT_FIELD] = prompts_responses
    
    return assistants

def preparedata(mode):
    print("Preparing data for - ", mode, "...")
    dataset_df = load_local_dataset(config.DATASET_PATH)
    prompts_responses = prepare_prompts_responses(dataset_df)
    prompts_responses_dataset = datasets.Dataset.from_pandas(prompts_responses)
    return prompts_responses_dataset
