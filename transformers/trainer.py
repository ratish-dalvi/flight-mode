import time
import argparse
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer import Transformer
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  


class C4Dataset(IterableDataset):
    def __init__(self, seq_len, tokenizer_name="gpt2", split='train'):
        # Load dataset information for streaming
        self.dataset = load_dataset("c4", "en", split=split, streaming=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        # Sequence length
        self.seq_len = seq_len

    def __iter__(self):
        # Iterator over the dataset
        for data in self.dataset:
            # Tokenize text on-the-fly
            if len(data['text']) < self.seq_len:
                continue
            tokens = self.tokenizer.encode(data['text'], add_special_tokens=False)

            # Split into sequences of the desired length
            for i in range(0, len(tokens), self.seq_len):
                sequence = tokens[i:i+self.seq_len+1]
                if len(sequence) == self.seq_len + 1:
                    input_ids = sequence[:-1]
                    target_ids = sequence[1:]
                    yield torch.tensor(input_ids), torch.tensor(target_ids)
    

class Trainer:
    def __init__(self, config):
        
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize C4Dataset
        self.train_dataset = C4Dataset(config['context_length'], split="train")
        print("Initialized dataset")
        
        self.model = Transformer(
            config['embedding_size'], self.train_dataset.vocab_size,
            config['context_length'], config['num_layers'], config['dropout'],
            config['mult'], config['num_heads'], self.device
        )
        self.model = self.model.to(self.device)
        self.writer = SummaryWriter()
        

    def load_model(self, load_path):
        """Load the model from the specified path."""
        state_dict = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")
        
    def save_model(self, save_path):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), save_path)
        
    def run(self):

        # setup the optimizer
        print(f"Model: {self.model}")
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_parameters2 = sum(p.numel() for p in self.model.parameters())
        
        print(f"The Transformer model has {num_parameters / 1000000: .2f} M trainable parameters.")

        opt = torch.optim.Adam(lr=self.config["lr"], params=self.model.parameters())

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset, pin_memory=True, batch_size=self.config["batch_size"])

        self.model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
        
            x, y = batch
            
            # forward the model
            logits, self.loss = self.model(x, y)

            # backprop and update the parameters
            # self.model.zero_grad(set_to_none=True)
            opt.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            opt.step()

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.iter_num % self.config["save_every"] == 0:
                self.save_model(f'model_checkpoint_{self.iter_num}.pt')
                self.evaluate()
                
            # Logging to TensorBoard
            self.writer.add_scalar('Loss/train', self.loss.item(), self.iter_num)
            
            # Optionally, add code here to compute and log evaluation loss
            if self.iter_num % self.config["print_every"] == 0:
                print(f"Iteration: {self.iter_num}, Training Loss: {self.loss.item()}")


    def evaluate(self):
        """Evaluate the model on a couple of set prompts"""
        self.model.eval()

        print("\n")
        with torch.no_grad():  # Disable gradient computation
            for prompt in self.config["evaluation_prompts"]:
                generated_text = self.generate_text(prompt)
                print(generated_text)
                print("\n")

        # Set the model back to training mode
        self.model.train()
                
    def generate_text(self, prompt, max_tokens=50, temperature=1):
        """
        Generate text given some initial text.
        """
        # Encode the initial text
        input_ids = self.train_dataset.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)

        st = time.time()
        # Generate additional tokens
        generated_ids = self.model.generate(input_ids, max_tokens, temperature=temperature)
        print(f"Text generated at {generated_ids.shape[1]/(time.time() - st):.1f} tokens/second")
        
        # Decode the generated tokens to text
        generated_text = self.train_dataset.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return generated_text

    
def parse_args(default_config):
    parser = argparse.ArgumentParser(description='Train a Transformer model from scratch')
    for key, value in default_config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser.parse_args()

def main():
    # Default configuration
    default_config = {
        "embedding_size": 384,
        "context_length": 512,
        "num_layers": 12,
        "dropout": 0,
        "mult": 4,
        "num_heads": 12,
        "lr": 0.0003,
        "batch_size": 16,
        "num_workers": 8,
        "grad_clip": 1,
        "print_every": 200,
        "save_every": 5000,
        "evaluation_prompts": ["Once upon a time, ", "The president of the united states is", "The best thing about"]
    }

    args = parse_args(default_config)

    # Override defaults with any command-line arguments
    config = {key: getattr(args, key) for key in default_config}

    training_instance = Trainer(config)
    training_instance.run()

if __name__ == "__main__":
    main()
