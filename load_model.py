from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_and_tokenizer(config):
    model_name = config['model']['name']
    num_labels = config['model']['num_labels']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return model, tokenizer

if __name__ == "__main__":
    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config)
    print("Model and tokenizer loaded successfully.")