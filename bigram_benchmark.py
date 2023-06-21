import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_batch, estimate_loss

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 250
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
seed = 42
# ------------

torch.manual_seed(seed)
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


class BigramLanguageModel(nn.Module):
    """Simple Bigram Language model based only on token embeddings.
    
    The model learns to predict next token based on current token only.
    """
    
    def __init__(self, vocab_size):
        """Initializes the instance based on vocabulary size.

        Args:
            vocab_size (int): number of tokens in the entire dictionary
        """
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        """Forward pass from input to output.

        Args:
            idx (torch.tensor): tensor with token ids contained in sequence -> (B, T)
            targets (torch.tensor, optional): next token to be predicted. Defaults to None.

        Returns:
            torch.tensor: predictions and loss from the forward pass
        """
        # idx and targets are both (B,T) tensor of integers
        # we create a table of embs contained in token_embedding_table searching by idx
        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # prepare output and target for cross-entropy -> join all batches into one matrix
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # we evaluate all embeddings of all tokens of all batches at once
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate a sequence of tokens

        Args:
            idx (torch.tensor): tensor with token ids contained in sequence -> (B, T)
            max_new_tokens (int): number of tokens to generate

        Returns:
            torch.tensor: input sequence of token with the new predicted token added
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(
            train_data, 
            val_data, 
            model,
            batch_size, 
            block_size,
            eval_iters
        )
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, batch_size, block_size)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))