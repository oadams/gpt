import math 
from typing import Dict, List, Tuple, Optional

from jaxtyping import Float, Integer
import torch
from torch import Tensor
import torch.nn
import torch.random
from torch.utils.tensorboard import SummaryWriter
import tqdm

writer = SummaryWriter()

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

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


def create_batch(batch_size: int, context_length: int, split: str) -> Tuple[Float[Tensor, 'B T C'], Float[Tensor, 'B T C']]:
    """ Create a random batch of examples. This consists of two tensors of the
    same shape, x and y. y is just x offset by one timestep so that each
    position in y corresponds to the same position in x but one timestep
    further. For now there are only two splits, 'train' and 'test'.
    """

    if split == 'train':
        text_ids = train
    elif split == 'test':
        text_ids = test
    else:
        raise ValueError()
    rand_starts: Float[Tensor, '...'] = torch.randint(len(text_ids) - context_length, (batch_size,))
    x = torch.tensor([text_ids[rand_start:rand_start+context_length] for rand_start in rand_starts.tolist()]).to(device)
    y = torch.tensor([text_ids[rand_start+1:rand_start+context_length+1] for rand_start in rand_starts.tolist()]).to(device)
    return x, y


def estimate_loss(model: torch.nn.Module, eval_iters: int, batch_size: int, context_length: int) -> Dict[str, Float[Tensor, '...']]:
    """ Estimates the loss of the model by randomly sampling `eval_iters`
    batches and averaging the loss. It's not an exhaustive evaluation of the
    test set.
    """

    model.eval()
    result: Dict[str, Float[Tensor, '...']] = {}
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = create_batch(batch_size, context_length, split)
            _, loss = model(x, y)
            losses[iter] = loss.item()
        result[split] = losses.mean()
    return result


class Attention(torch.nn.Module):
    """ Single head attention module. """

    def __init__(self, input_dim: int, output_dim: int, dropout: float, context_length: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Wk = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wq = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wv = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        K: Float[Tensor, 'B T H'] = self.Wk(x)
        Q: Float[Tensor, 'B T H'] = self.Wq(x)
        V: Float[Tensor, 'B T H'] = self.Wv(x)
        wei = Q @ K.transpose(1, 2) / math.sqrt(K.shape[-1])
        T = x.shape[1]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        wei = self.dropout(wei)
        return wei @ V


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int, dropout: float, context_length: int) -> None:
        super().__init__()
        assert output_dim % num_blocks == 0
        self.heads = torch.nn.ModuleList(Attention(input_dim, int(output_dim / num_blocks), dropout, context_length) for _ in range(num_blocks))
        self.proj = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        head_outs = [head(x) for head in self.heads]
        return self.proj(torch.cat(head_outs, dim=-1))


class TransformerLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int, dropout: float, context_length: int) -> None:
        super().__init__()
        self.mh_attention = MultiHeadAttention(input_dim, output_dim, num_blocks, dropout, context_length)
        self.ff = torch.nn.Linear(output_dim, 4*output_dim)
        self.proj = torch.nn.Linear(4*output_dim, output_dim)
        self.layernorm1 = torch.nn.LayerNorm(output_dim)
        self.layernorm2 = torch.nn.LayerNorm(output_dim)
        self.mh_dropout = torch.nn.Dropout(dropout)
        self.ff_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        x = x + self.mh_dropout(self.mh_attention(self.layernorm1(x)))
        x = x + self.ff_dropout(self.proj(torch.nn.functional.relu(self.ff(self.layernorm2(x)))))
        return x


class GPT(torch.nn.Module):
    def __init__(self, vocab: List[str], hdim: int, context_length: int, num_layers: int, dropout: float):
        super().__init__()
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(len(vocab), hdim)
        self.pos_embedding = torch.nn.Embedding(context_length, hdim)
        self.final_proj = torch.nn.Linear(hdim, len(vocab))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.layers = torch.nn.ModuleList([TransformerLayer(hdim, hdim, 2, dropout, context_length) for _ in range(num_layers)])
        self.layernorm = torch.nn.LayerNorm(hdim)

    def forward(self, x: Integer[Tensor, 'B T'], y: Optional[Integer[Tensor, 'B T']] = None) -> Tuple[Float[Tensor, 'B T C'], Optional[Float[Tensor, '...']]]:
        T = x.shape[-1]
        x = self.embedding(x) + self.pos_embedding(torch.arange((T), device=device))
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)
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

batch_size = 64
context_length = 128 
lr = 3e-4
hidden_size = 256
num_layers = 4
dropout = 0.2
gpt = GPT(vocab, hidden_size, context_length, num_layers, dropout).to(device)
total_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
print(f'{total_params=}')
print((p, p.numel()) for p in gpt.parameters() if p.requires_grad)

context = torch.tensor(c2i('\n')).view(1, 1).to(device)
print(context)
print(context.shape)
print(estimate_loss(gpt, 500, batch_size, context_length))
print('No training: ', ''.join(i2c(x) for x in gpt.generate(context, 100, context_length)[0].tolist()))
optim = torch.optim.AdamW(gpt.parameters(), lr=lr)

for step in tqdm.tqdm(range(20000)):
    gpt.zero_grad()
    batch = create_batch(batch_size, context_length, 'train')
    logits, loss = gpt(*batch)
    loss.backward()
    optim.step()
    if step % 500 == 0:
        result = estimate_loss(gpt, 500, batch_size, context_length)
        writer.add_scalar('Loss/train', result['train'], step)
        writer.add_scalar('Loss/test', result['test'], step)
print(estimate_loss(gpt, 500, batch_size, context_length))
print(''.join(i2c(x) for x in gpt.generate(context, 1000, context_length)[0].tolist()))