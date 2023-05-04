import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['conversations']

def save_conversations_to_txt(conversations, output_file):
    with open(output_file, 'w') as f:
        for conversation in conversations:
            for message in conversation:
                f.write(message + "\n")
            f.write("\n")

def main():
    # Set the model and tokenizer
    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    # Load and preprocess the dataset
    dataset_path = "path/to/your/facebook/messenger/dataset.json"
    conversations = load_dataset(dataset_path)
    txt_file = "conversations.txt"
    save_conversations_to_txt(conversations, txt_file)

    # Create a dataset from the text file
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=txt_file,
        block_size=128,
    )

    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set training arguments
    output_dir = "fine_tuned_model"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
