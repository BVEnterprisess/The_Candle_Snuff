from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import yaml
from load_model import load_model_and_tokenizer
from data_prep import preprocess_data

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load test data
    test_dataset = load_dataset('csv', data_files=config['data']['test_path'])['train']
    test_dataset = preprocess_data(test_dataset, tokenizer)
    
    # Training arguments for evaluation
    training_args = TrainingArguments(
        output_dir=config['output']['model_save_path'],
        per_device_eval_batch_size=config['training']['batch_size'],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )
    
    # Evaluate
    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()