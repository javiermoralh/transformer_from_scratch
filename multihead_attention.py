import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_batch, estimate_loss

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 32
n_heads = 4
head_size = n_emb//n_heads
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
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


class Head(nn.Module):
    """Self-attention head"""

    def __init__(self, head_size):
        """Initializes the instance based on head size.

        Args:
            head_size (int): key, query and value matrices column dimension
        """
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangle matrix

    def forward(self, x):
        """Forward pass from head input to head output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: tensor with encoded token embeddings contained in sequence -> (B, T, hs)
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""

    def __init__(self, num_heads, head_size):
        """Initializes the instance based on number of head and head size.

        Args:
            num_heads (int): number of self-attention heads
            head_size (int): key, query and value matrices column dimension
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        """Forward pass from head multihead input to multihead output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: tensor with encoded token embeddings contained in sequence -> (B, T, n_emb)
        """
        # all self-attention heads outputs are concatenated: h(x) (B, T, hs)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # out (B, T, n_emb)
        return out


class FeedFoward(nn.Module):
    """Simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        """Initializes the instance based on embedding size.

        Args:
            n_embd (int): size of input embedding
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass from input to output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: mlp output tensor -> (B, T, n_embd)
        """
        print(self.net(x).shape)
        return self.net(x)


class MultiheadAttentionLanguageModel(nn.Module):
    """Language model based on multi-head self-attention.
    
    The model learns to predict next token using multi-head self-attention.
    """
    
    def __init__(self, vocab_size):
        """Initializes the instance based on vocabulary size.

        Args:
            vocab_size (int): number of tokens in the entire dictionary
        """
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.sa_heads = MultiHeadAttention(n_heads, head_size)  # 4 heads of 8-dimnesional self-attention
        self.ffwd = FeedFoward(n_emb)  # to add more complexity, not just a linear output
        self.lm_head = nn.Linear(n_emb, vocab_size)
    
    def forward(self, idx, targets=None):
        """Forward pass from input to output.

        Args:
            idx (torch.tensor): tensor with token ids contained in sequence -> (B, T)
            targets (torch.tensor, optional): next token to be predicted. Defaults to None.

        Returns:
            torch.tensor: predictions and loss from the forward pass
        """
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # rearange matrix to sum -> (T,C)
        x = token_emb + pos_emb  # (B,T,C) with this we dont'just have the token identity, we also have the position
        x = self.sa_heads(x)  # apply heads of self-attention
        x = self.ffwd(x)  # apply feed forward block
        logits = self.lm_head(x)  # logits: outputs of the linear module (B,T,vocab_size)
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
        # idx is (B, T) array of indices in the current contextsy
        for _ in range(max_new_tokens):
            # crop (recortar) idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = MultiheadAttentionLanguageModel(vocab_size)
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
