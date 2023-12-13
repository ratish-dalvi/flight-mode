import time
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer import Transformer
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader



class Enwik8Dataset(IterableDataset):
    def __init__(self, seq_len, tokenizer_name="gpt2"):
        # Load dataset information for streaming
        self.dataset = load_dataset("enwik8", split='train', streaming=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)

        # Sequence length
        self.seq_len = seq_len

    def __iter__(self):
        # Iterator over the dataset
        for data in self.dataset:
            # Tokenize text on-the-fly
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

        # Initialize Enwik8Dataset
        self.train_dataset = Enwik8Dataset(config['context_length'], tokenizer_name="gpt2")
        print("Initialized dataset")
        
        self.model = Transformer(
            config['embedding_size'], self.train_dataset.vocab_size, config['context_length'],
            config['num_layers'], config['dropout'], config['mult'], config['num_heads']
        )

    def run(self):

        # setup the optimizer
        print(f"Model: {self.model}")
        print(f"Model parameters: {self.model.parameters()}")

        opt = torch.optim.Adam(lr=config["lr"], params=self.model.parameters())

        # self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            # sampler=torch.utils.data.RandomSampler(
            #     self.train_dataset, replacement=True, num_samples=int(1e10)),
            # shuffle=False,
            pin_memory=True,
            batch_size=config["batch_size"],
            # num_workers=config["num_workers"],
        )

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
            print(f"Shape of input: {x.shape}")
                    
            # forward the model
            logits, self.loss = self.model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

# Example usage
# model and config should be defined according to your requirements
config = {
    "embedding_size": 128,
    "context_length": 512,
    "num_layers": 4,
    "dropout": 0,
    "mult": 4,
    "num_heads": 4,
    "lr": 0.0001,
    "batch_size": 16,
    "num_workers": 8
}

training_instance = Trainer(config)
training_instance.run()

