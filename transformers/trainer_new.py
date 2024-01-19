from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from time import time


def load_and_split_dataset(tokenizer, seq_len, split_ratio=0.1, dataset_percent=1):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len, padding='max_length')

    split = f'train[:{dataset_percent}%]' if dataset_percent is not None else 'train'
    dataset = load_dataset("openwebtext", split=split)    
    # Tokenize the dataset. For full dataset, takes ~30 mins the first time
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, batch_size=10000, remove_columns=['text'])
    
    print(f"Full tokenized dataset: {tokenized_dataset}")    
    print("\nSplitting dataset into train/eval ...") # for full dataset, takes ~5 mins the first time
    
    train_test_split = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    return train_dataset, eval_dataset


# Custom Data Collator
class CustomDataCollatorForLanguageModeling:
    def __call__(self, examples):
        # Collate batches of examples
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])

        labels = input_ids.clone()
        labels[:, :-1] = labels[:, 1:]
        labels[:, -1] = -100  # Ignore the computation loss for the last position

        return {"input_ids": input_ids, "labels": labels}

    
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load and split the dataset
st = time()
train_dataset, eval_dataset = load_and_split_dataset(tokenizer, seq_len=256, dataset_percent=1)
print(f"Train: {train_dataset}")
print(f"Eval: {eval_dataset}")
print(f"Datasets created/loaded in {time()-st:.1f} seconds")

# Initialize the custom data collator
data_collator = CustomDataCollatorForLanguageModeling()


config = {
    "embedding_size": 384,
    "context_length": 512,
    "num_layers": 12,
    "dropout": 0,
    "mult": 4,
    "num_heads": 12,
    "lr": 0.0003,
    "batch_size": 16,
    "num_workers": 8,
    "grad_clip": 1
}

model = Transformer(
    config['embedding_size'], self.train_dataset.vocab_size,
    config['context_length'], config['num_layers'], config['dropout'],
    config['mult'], config['num_heads'], self.device
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./model_output")
