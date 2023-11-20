import torch.nn
import torch.random
from typing import List


with open('crime_and_punishment.txt') as f:
    text = f.read()
print(len(text))

vocab = sorted(set(text))

def c2i(c: str) -> int:
    return vocab.index(c)

def i2c(i: int) -> str:
    return vocab[i]

torch.random.manual_seed(1337)

# Let's construct our training examples.
# We want to take our text, some block size, and generate a random sample.
# How did Karpathy's approach ensure the whole data was used in a given epoch?
# I seem to recall him randomly choosing an offset. Well let's just go with that for now
# and update it later when we want to do training
def create_example(batch_size: int, block_size: int, text: str):
    x = torch.randint(len(text) - block_size, (batch_size,))
    print(x)

class GPT(torch.nn.Module):
    def __init__(self, vocab: List[str], hdim: int):
        self.vocab = vocab
        self.embedding = torch.nn.Embedding(len(vocab), hdim)
        # Now we need to make a loss.

create_example(4, 8, text)