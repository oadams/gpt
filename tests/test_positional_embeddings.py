import torch

from positional_embeddings import RotaryEmbedding
from rotary_embedding_torch import RotaryEmbedding as RefRotaryEmbedding


def test_rope():
    x = torch.ones((2, 11, 6))

    ref = RefRotaryEmbedding(dim=x.shape[-1])
    rope = RotaryEmbedding(dim=x.shape[-1])
    e = rope.rotate_queries_or_keys(x)
    e_ref = ref.rotate_queries_or_keys(x)
    print(e)
    print(e_ref)
    assert (e == e_ref).all()
