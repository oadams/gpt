import math 
from typing import Dict, List, Tuple, Optional

from jaxtyping import Float, Integer
import torch
from torch import Tensor
import torch.nn
import torch.random

with open('crime_and_punishment.txt') as f:
    text = f.read()
print(len(text))

vocab = sorted(set(text))

def c2i(c: str) -> int:
    return vocab.index(c)

def i2c(i: int) -> str:
    return vocab[i]

text_ids = [c2i(c) for c in text]

torch.random.manual_seed(1337)

k = int(0.9*len(text_ids))
train = text_ids[:k]
test = text_ids[k:]

# Let's construct our training examples.
# We want to take our text, some block size, and generate a random sample.
# How did Karpathy's approach ensure the whole data was used in a given epoch?
# I seem to recall him randomly choosing an offset. Well let's just go with that for now
# and update it later when we want to do training
def create_batch(batch_size: int, block_size: int, split: str) -> Tuple[Float[Tensor, 'B T C'], Float[Tensor, 'B T C']]:
    if split == 'train':
        text_ids = train
    elif split == 'test':
        text_ids = test
    else:
        raise ValueError()
    rand_starts: Float[Tensor, '...'] = torch.randint(len(text_ids) - block_size, (batch_size,))
    x = torch.tensor([text_ids[rand_start:rand_start+block_size] for rand_start in rand_starts.tolist()])
    y = torch.tensor([text_ids[rand_start+1:rand_start+block_size+1] for rand_start in rand_starts.tolist()])
    return x, y

x, y = create_batch(4, 8, 'test')

def estimate_loss(model: torch.nn.Module, eval_iters: int, batch_size: int, block_size: int) -> Dict[str, Float[Tensor, '...']]:
    model.eval()
    result: Dict[str, Float[Tensor, '...']] = {}
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = create_batch(batch_size, block_size, split)
            _, loss = model(x, y)
            losses[iter] = loss.item()
        result[split] = losses.mean()
    return result


class Attention(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # What about weight initialization for these?
        self.Wk = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wq = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wv = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        K: Float[Tensor, 'B T H'] = self.Wk(x)
        Q: Float[Tensor, 'B T H'] = self.Wq(x)
        V: Float[Tensor, 'B T H'] = self.Wv(x)
        wei = Q @ K.transpose(1, 2) / math.sqrt(K.shape[-1])
        T = x.shape[1]
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        return torch.nn.functional.relu(wei) @ V

class GPT(torch.nn.Module):
    def __init__(self, vocab: List[str], hdim: int, context_length: int):
        super().__init__()
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(len(vocab), hdim)
        self.pos_embedding = torch.nn.Embedding(context_length, hdim)
        self.final_proj = torch.nn.Linear(hdim, len(vocab))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.attention = Attention(hdim, hdim)
        # Now we need to make a loss.

    def forward(self, x: Integer[Tensor, 'B T'], y: Optional[Integer[Tensor, 'B T']] = None) -> Tuple[Float[Tensor, 'B T C'], Optional[Float[Tensor, '...']]]:
        T = x.shape[-1]
        x = self.embedding(x) + self.pos_embedding(torch.arange((T)))
        x = self.attention(x)
        logits = self.final_proj(x)

        if y is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = self.loss_fn(logits, y)
            return logits, loss

    def generate(self, context: Integer[Tensor, 'B T'], max_output_length: int, context_length: int) -> Integer[Tensor, 'B T']:
        for _ in range(max_output_length):
            # Convert context into input IDs. This requires knowing context size.
            logits, _ = self(context[:, -context_length:]) # B T C
            logits = logits[:, -1, :]
            # Now find the word closest to this logit. Question: How is that done? Answer: it actually is sampling just from a multinomial over the probs.  
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            # Then feed those words as context in. Once you get to length `context_size` start doing a sliding window.
            context = torch.cat((context, idx), dim=1)
        return context

context_length = 8
gpt = GPT(vocab, 32, context_length)
gpt(x, None)
context = torch.tensor(c2i('\n')).view(1, 1)
print(context)
print(context.shape)
print(estimate_loss(gpt, 500, 4, context_length))
print('No training: ', ''.join(i2c(x) for x in gpt.generate(context, 100, context_length)[0].tolist()))
optim = torch.optim.AdamW(gpt.parameters(), lr=1e-3)

for step in range(10000):
    gpt.zero_grad()
    batch = create_batch(4, context_length, 'train')
    logits, loss = gpt(*batch)
    loss.backward()
    optim.step()
print(estimate_loss(gpt, 500, 4, 8))
print(''.join(i2c(x) for x in gpt.generate(context, 100, context_length)[0].tolist()))