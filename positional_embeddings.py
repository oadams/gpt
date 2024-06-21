import math
import torch

from config import config


class RoPE2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 2 * math.pi / 10

    def forward(self, x):
        """Just a 2D rope embedding application to get a better understanding

        x is BxTxC
        """

        # For each time step we need to rotate by a different amount, so we really want a different matrix for each timestep
        T = x.shape[-2]
        results = []
        for m in range(T):
            r = torch.tensor(
                [
                    [math.cos(m * self.theta), -math.sin(m * self.theta)],
                    [math.sin(m * self.theta), math.cos(m * self.theta)],
                ]
            )
            results.append(r @ x[m, :])
        result = torch.stack(results)
        return result


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO Use this info and optimize the forward pass
        # self.dim = dim

    def rotate_queries_or_keys(self, x):
        # This is just to be consistent with the lucid rains interface.
        return self.forward(x)

    def forward(self, x):
        # Our vector x will be broken
        # into groups of two elements, which will be rotated.
        # Each pair will be rotated at a different rate. The first
        # vector is rotated quickly, the second pair slower, the
        # third pair slower still and so forth.
        # The result is that the dot product between some q and some k
        # will tend to be higher if they are further apart.
        # Moreover, the dot product between q & k will be the same
        # regardless of their absolute position as long as their relative
        # distance is the same.
        b, t, d = x.shape
        assert d % 2 == 0
        # First step is to make our theta, which comes
        # in identical pairs.

        i = torch.arange(1, d // 2 + 1, device=x.device)
        # RoPE authors took this theta setting from Vaswani 2017,
        # because it allowed for the long-term decay property
        theta = 10000 ** (-2 * (i - 1) / d)
        # This just creates our pairs of identical theta from unique thetas
        # by stacking theta side-by-side with itself and changing the view.
        # E.g.:
        # [t1, t2, t3] -> [t1, t1, t2, t2, t3, t3]
        theta = torch.cat((theta[:, None], theta[:, None]), dim=-1).view(-1)
        results = []
        for m in range(t):
            # Here m represents our timestep. Larger values of m get rotated more.
            # We grab the embedding at timestep m.
            xm = x[:, m, :]
            # The rest of this block is applying equation (34) from the Su et al paper
            # Term 1 is the product on the left and straightforwardly maps to what you see in the paper
            term1 = torch.cos(m * theta) * xm
            # The next bit is the second term (the product on the right)
            # It's fiddly because we got [-x2 x1 -x4 x3 ... -xd x_{d-1} ].
            # The approach below is just some jiggery-pokery to get those
            # xs arranged correctly.
            xm = xm.view((b, d // 2, 2))
            xm = torch.stack((xm[:, :, 1], xm[:, :, 0])).T
            xm[:, :, 0] *= -1
            xm = xm.reshape(b, d)
            # Now we get that second product
            term2 = torch.sin(m * theta) * xm
            results.append(term1 + term2)
        # Stack all our embeddings back together again.
        result = torch.stack(results, dim=1)
        return result


if config["lucidrains_rope"]:
    from rotary_embedding_torch import RotaryEmbedding
