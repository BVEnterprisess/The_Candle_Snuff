import pandas as pd
from datasets import Dataset
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

def preprocess_data(df, tokenizer, max_length=512):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    config = load_config()
    # Assuming data is in CSV with 'text' and 'label' columns
    train_df, val_df, test_df = load_data(config['data']['train_path'], config['data']['val_path'], config['data']['test_path'])
    
    # Note: Tokenizer will be loaded in training script
    print("Data loaded and ready for preprocessing.")