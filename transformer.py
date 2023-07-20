import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
from AE import AE

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 20 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)


# Train and test splits
data = np.array((100, 2, n_embd))
train_data = data[:80]
val_data = data[80:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key_n_query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(20, 20)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        embeddings = x[0]
        #print("Embeddings have the size of : " , embeddings.shape)
        x = x[1]
        B,T,C = x.shape
        #print("X shape is ", x.shape, "")
        k = self.key_n_query(embeddings)   # (B,T,hs)
        q = self.key_n_query(embeddings) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #print("Wei shape is ", wei.shape)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pos_emb, x):
        out = torch.cat([h((x, pos_emb)) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        pos_emb = x[0]
        x = x[1]
        x = x + self.sa(self.ln1(pos_emb), self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = (x, pos_emb)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, model:AE):
        super().__init__()
        self.position_embedding = nn.Linear(2, 32)
        self.model = model
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, n_embd)
        self.positional_recurcer = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid(),
        )
        self.pos_prog = nn.Conv1d(1,32, kernel_size=64, stride=32)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, locs, targets=None):
        # X will be ready, B is the number of sequences in the batch, T is the length of each sequence
        # While C is the length of the representation of each token (the embedding size)
        B, T, C = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = idx # (B,T,C)
        # generate a unique positional embedding for each token in the sequence based on the location
        pos_emb = self.position_embedding(locs)
        pos_emb[:, 0, :] = self.positional_recurcer(torch.cat([torch.zeros((B, 32)).to(self.model.device), pos_emb[:, 0, :]], dim = 1))
        for i in range(1, T):
            pos_emb[:, i, :] = self.positional_recurcer(torch.cat([pos_emb[:, i - 1, :], pos_emb[:, i, :]], dim=1))
        x = self.blocks((tok_emb,pos_emb)) # (B,T,C)
        x = self.ln_f(x[0]) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is not None:
            recon_batch = self.model.model.decode(logits.view(-1, 32))
            data = self.model.model.decode(targets.view(-1, 32))
            losses = self.model.loss_function(recon_batch, data)
            loss = losses
        else:
            loss = None

        return logits, loss

    def policy(self) :
        # generate two random numbers between 0 and 640 - 64
        dx = random.randint(0, 640 - 64)
        dy = random.randint(0, 640 - 64)
        return np.array([dx, dy], dtype=np.float32)
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
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