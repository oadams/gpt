"""Tensor operations and autograd"""

import torch


class GradError(Exception):
    pass


class Tensor:
    def __init__(
        self, data, dtype=torch.float32, device="cpu", requires_grad=False, is_leaf=True
    ):
        self.data = torch.tensor(data, dtype=dtype, device=device, requires_grad=False)
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad
        self.grad = None
        self.backward = None

    def shape(self):
        return self.data.shape

    def __repr__(self):
        data_str = repr(self.data).replace("tensor", "data")
        return data_str

    def __add__(self, other):
        result_data = self.data + other.data
        result = Tensor(
            result_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            is_leaf=False,
        )

        def backward(dL_dthis):
            if dL_dthis.shape() != result_data.shape:
                raise GradError(
                    "dL_dthis isn't of the right shape. It needs to match the shape of the tensor we're calling backward on."
                )
            # The reason dL_dthis is the gradient is because the gradient of result with respect to self is just a tensor of ones (it's addition) so overall the dL/d_self
            # will just be dL/d_this element-wise times ones.
            if self.requires_grad:
                if self.grad is not None:
                    self.grad = self.grad + dL_dthis.data
                else:
                    self.grad = dL_dthis.data
            if other.requires_grad:
                if other.grad is not None:
                    other.grad = self.grad + dL_dthis.data
                else:
                    other.grad = dL_dthis.data

        result.backward = backward

        return result
