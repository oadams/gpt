"""
A basic implementation of a decoder-only transformer.

Uses jaxtyping to annotate the shapes of tensors. The convention here is:
    - B = batch dimension
    - T = 'time' dimension. Corresponds to the length of the sequence.
    - C = 'channel' dimension. Corresponds to the size of the embedding, basically how many tokens are in the vocabulary.
    - H = hidden dimension. Corresponds to the size of the hidden state of the transformer.
"""
import argparse
import math 
from typing import Dict, List, Tuple, Optional

from beartype import beartype as typechecker
from jaxtyping import Float, Integer, jaxtyped
import platform
import tiktoken
import torch
from torch import Tensor
import torch.nn
import torch.random
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from optimizers import SGD, RMSProp, Adam, AdamW
from normalization import LayerNorm

LayerNorm = torch.nn.LayerNorm


def get_device():
    """ Make use of the best hardware we have available. Leverage Apple Silicon
    GPUs with Metal Performance Shaders (MPS) on Macs, and CUDA on other
    systems.  """
    # Check if the machine is a Mac
    if platform.system() == 'Darwin':
        # Check for MPS availability (requires PyTorch 1.12.0 or later)
        if torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) on Mac")
            return torch.device("mps")
        else:
            print("MPS not available, using CPU on Mac")
            return torch.device("cpu")
    else:
        # Check for CUDA availability on non-Mac systems
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device("cuda")
        else:
            print("CUDA not available, using CPU")
            return torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', default='crime_and_punishment.txt', help='Plain UTF-8 text file as training data. Gets split 90/10')
parser.add_argument('--device', default=get_device())
parser.add_argument('--tokenizer', default='char', choices=['char', 'bpe'])
parser.add_argument('--random_seed', default=0)
parser.add_argument('--train_fraction', default=0.9)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--context_length', default=128)
parser.add_argument('--lr', default=3e-4, help='Learning rate')
parser.add_argument('--hidden_size', default=256)
parser.add_argument('--num_layers', default=6)
parser.add_argument('--num_heads', default=4)
parser.add_argument('--dropout', default=0.2)
parser.add_argument('--n_estimate_steps', default=100, help='The number of steps to take in loss estimation (it is probabilistic)')
parser.add_argument('--n_gen_tokens', type=int, default=500, help='The number of tokens to generate during inference.')
parser.add_argument('--train_steps', default=10000)
parser.add_argument('--save_eval_every_n_steps', default=100)
parser.add_argument('--model_path')#, default='model_1000.pth')
parser.add_argument('--generate_only', default=False, action='store_true')
parser.add_argument('--payg', default=True, help='Print as you go')
parser.add_argument('--decode_greedy', default=False, action='store_true')
parser.add_argument('--decode_topk', default=None, type=int)
parser.add_argument('--prompt', default='\n', type=str, help='A prompt for the model')

args = parser.parse_args()

writer = SummaryWriter()

torch.random.manual_seed(args.random_seed)

if isinstance(args.device, str):
    args.device = torch.device(args.device)

with open(args.corpus, encoding='utf-8') as f:
    text = f.read()
print(f'Corpus length in Unicode codepoints: {len(text)}')

class CharTokenizer:
    """ A simple character-level tokenizer. Uses a similar interface to the tiktoken BPE encoders
    so that they can be used interchangeably.
    """
    def __init__(self, text: str) -> None:
        self.vocab = sorted(set(text))
        self.n_vocab = len(self.vocab)

    def c2i(self, c: str) -> int:
        return self.vocab.index(c)

    def i2c(self, i: int) -> str:
        return self.vocab[i]

    def encode(self, text: str) -> List[int]:
        return [self.c2i(c) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.i2c(i) for i in ids)

if args.tokenizer == 'bpe':
    # For now we just use GPT-2's BPE tokenizer. This has a ~50k vocabulary, so it's pretty big. Unless your model is 
    # quite big, you'll want to use a smaller vocabulary otherwise the embedding and final projection matrices will
    # be the bulk of the model's parameters.
    enc = tiktoken.get_encoding('gpt2')
elif args.tokenizer == 'char':
    enc = CharTokenizer(text)
else:
    raise ValueError("`--tokenizer` argument must be either 'char' or 'bpe'.")

text_ids = enc.encode(text)
print(f'Length of corpus in tokens: {len(text_ids)}')

k = int(args.train_fraction*len(text_ids))
train = text_ids[:k]
test = text_ids[k:]


@jaxtyped(typechecker=typechecker)
def create_batch(batch_size: int, context_length: int, split: str) -> Tuple[Integer[Tensor, 'B T'], Integer[Tensor, 'B T']]:
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
        raise ValueError('Split must be either "train" or "test"')
    rand_starts: Float[Tensor, 'B'] = torch.randint(len(text_ids) - context_length, (batch_size,))
    x = torch.tensor([text_ids[rand_start:rand_start+context_length] for rand_start in rand_starts.tolist()]).to(args.device)
    y = torch.tensor([text_ids[rand_start+1:rand_start+context_length+1] for rand_start in rand_starts.tolist()]).to(args.device)
    return x, y


@jaxtyped(typechecker=typechecker)
def estimate_loss(model: torch.nn.Module, eval_iters: int, batch_size: int, context_length: int) -> Dict[str, float]:
    """ Estimates the loss of the model by randomly sampling `eval_iters`
    batches and averaging the loss. It's not an exhaustive evaluation of the
    test set.
    """

    model.eval()
    result = {}
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = create_batch(batch_size, context_length, split)
            _, loss = model(x, y)
            losses[iter] = loss.item()
        result[split] = losses.mean().item()
    return result


class Attention(torch.nn.Module):
    """ Single head attention module. """

    def __init__(self, input_dim: int, output_dim: int, dropout: float, context_length: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Why do we disable biases for these linear layers?
        # One Answer: Our Q,K,V are just linear projections of the input, so we
        # don't need an extra bias.
        self.Wk = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wq = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.Wv = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        # register_buffer just stores this tensor in the model not as a parameter, so the optimizer
        # won't do anything with them.
        # tril is a triangular lower matrix and it's what we use to mask out the future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        K: Float[Tensor, 'B T H'] = self.Wk(x)
        Q: Float[Tensor, 'B T H'] = self.Wq(x)
        V: Float[Tensor, 'B T H'] = self.Wv(x)
        # For large dimensions H, the Q-K dot products can get very large, which means when we go
        # to compute the softmax we get small gradients. Dividing by sqrt(H) helps mitigate this.
        wei = Q @ K.transpose(1, 2) / math.sqrt(K.shape[-1])
        T = x.shape[1]
        # The magic to mask out future tokens. We set the future tokens to -inf so that when we
        # do a softmax they get a probability of 0.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        wei = self.dropout(wei)
        return wei @ V


class MultiHeadAttention(torch.nn.Module):
    """ Multihead attention is just a bunch of single head attentions concatenated together before a final projection.

    We enforce output dim to be divisible by number of heads and then for our
    final projection we use a square matrix. I don't think the transformer
    necessarily requires this. You could have as many heads of whatever shape
    you wwant and then concat and just project them to whatever hidden dimension
    you want, but we're just avoiding having one extra parameter.
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, dropout: float, context_length: int) -> None:
        super().__init__()
        assert output_dim % num_heads == 0
        self.heads = torch.nn.ModuleList(Attention(input_dim, int(output_dim / num_heads), dropout, context_length) for _ in range(num_heads))
        self.proj = torch.nn.Linear(output_dim, output_dim)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        head_outs = [head(x) for head in self.heads]
        return self.proj(torch.cat(head_outs, dim=-1))


class TransformerLayer(torch.nn.Module):
    """ A transformer 'block' or 'layer'.
    
    We use the pre-normalization variant of the transformer, which is popular.
    That is, we apply layer norm to the inputs before each of the attention and
    feed-forward layers.

    Note that we use one single dropout parameter in various places in this codebase.
    Since dropout is applied separately in different places you could have
    separate parameters but we've chosen to have just one to keep the
    hyperparameter space smaller.
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, dropout: float, context_length: int) -> None:
        super().__init__()
        self.mh_attention = MultiHeadAttention(input_dim, output_dim, num_heads, dropout, context_length)
        # In the original paper they said d_model is 512 and d_ff is 2048. This is why we use a 4x ratio here.
        self.ff = torch.nn.Linear(output_dim, 4*output_dim)
        self.proj = torch.nn.Linear(4*output_dim, output_dim)
        self.layernorm1 = LayerNorm(output_dim)
        self.layernorm2 = LayerNorm(output_dim)
        self.mh_dropout = torch.nn.Dropout(dropout)
        self.ff_dropout = torch.nn.Dropout(dropout)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, 'B T C']) -> Float[Tensor, 'B T H']:
        # x = x + y is a residiual connection.
        x = x + self.mh_dropout(self.mh_attention(self.layernorm1(x)))
        x = x + self.ff_dropout(self.proj(torch.nn.functional.gelu(self.ff(self.layernorm2(x)))))
        return x


class GPT(torch.nn.Module):
    def __init__(self, n_vocab: int, hdim: int, context_length: int, num_layers: int, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(n_vocab, hdim)
        # These are learned absolute positional embeddings. Many other options abound.
        self.pos_embedding = torch.nn.Embedding(context_length, hdim)
        self.final_proj = torch.nn.Linear(hdim, n_vocab)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.layers = torch.nn.ModuleList([TransformerLayer(hdim, hdim, num_heads, dropout, context_length) for _ in range(num_layers)])
        self.layernorm = LayerNorm(hdim)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Integer[Tensor, 'B T'], y: Optional[Integer[Tensor, 'B T']] = None) -> Tuple[Float[Tensor, 'B T C'], Optional[Float[Tensor, '']]]:
        T = x.shape[-1]
        x = self.embedding(x) + self.pos_embedding(torch.arange((T), device=args.device))
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
            return logits.view(B, T, C), loss

    @torch.no_grad()
    @jaxtyped(typechecker=typechecker)
    def generate(self, context: Integer[Tensor, 'B T'], max_output_length: int, context_length: int, payg=False, greedy=False, topk=None) -> Integer[Tensor, 'B max_output_length']:
        T = context.shape[-1]
        for _ in range(max_output_length-T):
            # Convert context into input IDs. This requires knowing context length the model can handle.
            logits, _ = self(context[:, -context_length:]) # B T C
            # Grab the final token. This is where we'll get the probabilities for the next token from.
            logits = logits[:, -1, :]
            # Now find the token closest to this logit.
            probs = torch.nn.functional.softmax(logits, dim=-1)
            if greedy:
                # Take the most likely token at each time step.
                idx = torch.argmax(probs, dim=1)[:, None]
            elif topk:
                # Take the top k tokens at each time step. We're basically just saying we never want to sample low probability tokens.
                top = torch.topk(probs, topk)
                # Renormalize probabilities and sample from them
                idx = torch.multinomial(top.values / top.values.sum(-1, keepdim=True), num_samples=1)
                # Convert indexes in top-k space into indexes in the full token vocab space.
                idx = top.indices.gather(dim=-1, index=idx)
            else:
                # Just do plain sampling from all possible tokens.
                idx = torch.multinomial(probs, 1)
            # Then feed those words as context in. Once you get to length `context_size` start doing a sliding window.
            context = torch.cat((context, idx), dim=1)
            if payg:
                B = context.shape[0]
                if B > 1:
                    raise ValueError("Can't print-as-you-go with batch size > 1. Either set batch size to 1 or turn off payg.")
                char: str = enc.decode([int(idx.item())])
                print(char, flush=True, end='')
        return context

gpt = GPT(enc.n_vocab, args.hidden_size, args.context_length, args.num_layers, args.dropout, args.num_heads).to(args.device)
if args.model_path is not None:
    gpt.load_state_dict(torch.load(args.model_path))
total_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
print('Model: ', gpt)
print(f'Total parameters in model: {total_params}')

print(estimate_loss(gpt, args.n_estimate_steps, args.batch_size, args.context_length))
#optim = SGD(gpt.parameters(), lr=.01, momentum_beta=0.9)
#optim = RMSProp(gpt.parameters(), lr=0.001, beta=0.99)
optim = AdamW(gpt.parameters(), lr=args.lr)
#optim = torch.optim.AdamW(gpt.parameters(), lr=args.lr)

if not args.generate_only:
    gpt.train()
    for step in tqdm.tqdm(range(args.train_steps)):
        gpt.zero_grad()
        batch = create_batch(args.batch_size, args.context_length, 'train')
        logits, loss = gpt(*batch)
        loss.backward()
        optim.step()
        if step % args.save_eval_every_n_steps == 0:
            torch.save(gpt.state_dict(), f'model_{step}.pth')
            result = estimate_loss(gpt, args.n_estimate_steps, args.batch_size, args.context_length)
            print(result)
            writer.add_scalar('Loss/train', result['train'], step)
            writer.add_scalar('Loss/test', result['test'], step)

    print(estimate_loss(gpt, args.n_estimate_steps, args.batch_size, args.context_length))
context = torch.tensor([enc.encode(args.prompt)]).to(args.device)
gpt.generate(context, args.n_gen_tokens, args.context_length, args.payg, args.decode_greedy, args.decode_topk)