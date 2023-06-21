import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import get_batch, estimate_loss

# PAPER:  https://arxiv.org/abs/1706.03762

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 384
n_head = 6
n_layer = 6
dropout = 0.3
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
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lowe triangle matrix
        self.dropout = nn.Dropout(dropout)

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
        wei = self.dropout(wei)
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
        self.proj = nn.Linear(n_emb, n_emb)  #  we apply a projection (linear transformation of the output)

    def forward(self, x):
        """Forward pass from head multihead input to multihead output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: tensor with encoded token embeddings contained in sequence -> (B, T, n_emb)
        """
        # all self-attention heads outputs are concatenated: h(x) (B, T, hs)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
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
            nn.Linear(n_embd, 4* n_embd),  # we multiply by 4 because it says so in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # we go back to original dimensionality
        )

    def forward(self, x):
        """Forward pass from input to output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: mlp output tensor -> (B, T, n_embd)
        """
        return self.net(x)


class Block(nn.Module):
    """Transformer block: arquitecture is defined in the paper."""

    def __init__(self, n_embd, n_head):
        """Initializes the instance based on embedding size and number of heads

        Args:
            n_embd (int): embedding dimension
            n_head (size): number of self-attention heads
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # for multihead attention output
        self.ln2 = nn.LayerNorm(n_embd)  # for feed forward output

    def forward(self, x):
        """Forward pass from input to output.

        Args:
            x (torch.tensor): tensor with token embeddings contained in sequence -> (B, T, C)

        Returns:
            torch.tensor: block output tensor -> (B, T, n_embd)
        """
        x = x + self.sa(self.ln1(x))  # we add residual connection with "x +"
        x = x + self.ffwd(self.ln2(x))   # we add residual connection with "x +"
        return x


class TransformerLanguageModel(nn.Module):
    """Language model based on transformer architecture.
    
    The model learns to predict next token using multi-head self-attention blocks.
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        """Initializes the instance based on vocabulary size.

        Args:
            vocab_size (int): number of tokens in the entire dictionary
        """
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
            Block(n_emb, n_head=n_head), # multihead-attention block with n_head heads
            Block(n_emb, n_head=n_head),
            Block(n_emb, n_head=n_head)
        )
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
        # first we get token embeddings and the the logits from the first head
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + pos_emb  # (B,T,C) with this we dont'just have the token identity, we also have the position
        x = self.blocks(x)  # apply heads of attention, feedfoward and layernorm of all blocks
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
        # idx is (B, T) array of indices in the current context
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

model = TransformerLanguageModel(vocab_size)
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

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
