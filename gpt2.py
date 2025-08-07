# this code is modified from Andrej Karpathy's video and code 
# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py


import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
torch.manual_seed(1337)


class TextData:
  def __init__(self, fin):
    with open(fin, 'r', encoding='utf-8') as f:
      self.text = f.read()
    self.chars = sorted(list(set(self.text)))
    self.vocab_size = len(self.chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(self.chars) }
    itos = { i:ch for i,ch in enumerate(self.chars) }
    self.encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
    n = int(0.9*len(self.data)) # first 90% will be train, rest val
    self.train_data = self.data[:n]
    self.val_data = self.data[n:]
  
  def get_batch(self, split):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
  

data = TextData('input.txt')
vocab_size = data.vocab_size
print (vocab_size)


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
        self.sa = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def get_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, x):
        B, T, _ = x.shape
        causal_mask = self.get_causal_mask(T, x.device)
        normed_x = self.ln1(x)
        attn_out, _ = self.sa(
            query=normed_x, 
            key=normed_x, 
            value=normed_x,
            attn_mask=causal_mask,
            key_padding_mask=None
           )
        x = x + attn_out
        normed_x = self.ln2(x)
        x = x + self.ffwd(normed_x)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = GPTLanguageModel()
model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def train(model, optimizer, loss_fn):
  model.train()
  xb, yb = data.get_batch('train')
  # evaluate the loss
  logits = model(xb)
  B, T, C = logits.shape
  logits = logits.view(B*T, C)
  targets = yb.view(B*T)
  loss = loss_fn(logits, targets)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  return loss.item()

@torch.no_grad()
def eval(model, loss_fn):
  model.eval()
  losses = torch.zeros(eval_iters)
  for k in range(eval_iters):
    X, y = data.get_batch('val')
    logits = model(X)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = y.view(B*T)
    loss = loss_fn(logits, targets)
    losses[k] = loss.item()
  model.train()
  return losses.mean()


# train model
for iter in range(max_iters):
  train_loss = train(model, optimizer, loss_fn)

  # every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0 or iter == max_iters - 1:
      val_loss = eval(model, loss_fn)
      print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data.decode(model.generate(context, max_new_tokens=500)[0].tolist()))