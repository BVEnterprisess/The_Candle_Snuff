import torch
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
    
    # Load and preprocess data
    train_dataset = load_dataset('csv', data_files=config['data']['train_path'])['train']
    val_dataset = load_dataset('csv', data_files=config['data']['val_path'])['train']
    
    train_dataset = preprocess_data(train_dataset, tokenizer)
    val_dataset = preprocess_data(val_dataset, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output']['model_save_path'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=config['output']['logs_path'],
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(config['output']['model_save_path'])
    tokenizer.save_pretrained(config['output']['model_save_path'])

if __name__ == "__main__":
    main()