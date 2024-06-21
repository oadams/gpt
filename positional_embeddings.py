import math
import torch


class RoPE2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 2 * math.pi / 10

    def forward(self, x):
        """Just a 2D rope embedding application

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


class RoPE(torch.nn.Module):
    def forward(self, x):
        b, t, d = x.shape
        assert d % 2 == 0
        i = torch.arange(1, int(d / 2) + 1, device=x.device)
        theta = 10000 ** (-2 * (i - 1) / d)
        theta = torch.cat((theta[:, None], theta[:, None]), dim=-1).view(-1)
        results = []
        for m in range(t):
            xm = x[:, m, :]
            term1 = torch.cos(m * theta) * xm
            xm = xm.view((b, d // 2, 2))
            xm = torch.stack((xm[:, :, 1], xm[:, :, 0])).T
            xm[:, :, 0] *= -1
            xm = xm.reshape(b, d)
            term2 = torch.sin(m * theta) * xm
            results.append(term1 + term2)
        result = torch.stack(results, dim=1)
        return result


if __name__ == "__main__":
    x = torch.ones((2, 11, 6))
    from rotary_embedding_torch import RotaryEmbedding

    ref = RotaryEmbedding(dim=x.shape[-1])
    rope = RoPE()
    e = rope(x)
    e_ref = ref.rotate_queries_or_keys(x.unsqueeze(0))
    print(e)
    print(e_ref)
