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

        def backward(dl_dr=None):
            """Run backpropagation on the sum of two tensors.

            Consider the addition `r = self + other`. `r` is the result. We
            are given dl/dr (the derivative of the loss with respect to the
            result r). Our goal is to:
                1. compute dl/dother and dl/dself.
                2. If self or other are not leaf nodes, then we call their backward() with dl/dother and dl/dself

            In the case of the sums, we have:
            - dr/dself = a tensor of ones: As each element of dself changes, the corresponding element of r changes proportionally
            - dl/dself = dl/dr * dr/dself: Each element of self can only affect the loss via the corresponding element of r, so we just take the hadamard
            product of dl/dr with the ones tensor, resulting in dl/dr tensor again.
            """
            if dl_dr is None:
                if len(result_data.shape) == 0:
                    dl_dr = Tensor(1)
                elif len(result_data.shape) == 1 and result_data.shape[0] == 1:
                    dl_dr = Tensor([1])
                else:
                    raise GradError(
                        "If tensor is not scalar then backwards must be called with a gradient of the same shape."
                    )
            if dl_dr.data.shape != result_data.shape:
                raise GradError(
                    "dL_dthis isn't of the right shape. It needs to match the shape of the tensor we're calling backward on."
                )
            # TODO Explain in clearer terms below how we arrived at this
            # gradient computation clearly. The clarity of the documentation and
            # story here is as important as the implementation.  The reason
            # dL_dthis is the gradient is because the gradient of result with
            # respect to self is just a tensor of ones (it's addition) so
            # overall the dL/d_self will just be dL/d_this element-wise times
            # ones.
            if self.requires_grad:
                self_grad = dl_dr.data
                if self.is_leaf:
                    if self.grad is not None:
                        self.grad = self.grad + self_grad
                    else:
                        self.grad = self_grad
                else:
                    self.backward(self_grad)
            if other.requires_grad:
                other_grad = dl_dr.data
                if other.is_leaf:
                    if other.grad is not None:
                        other.grad = other.grad + other_grad
                    else:
                        other.grad = other_grad
                else:
                    other.backward(other_grad)

        result.backward = backward

        return result

    def sum(self, dim=None, keepdim=False):
        result_data = self.data.sum(dim=dim, keepdim=keepdim)
        result = Tensor(result_data, requires_grad=self.requires_grad, is_leaf=False)

        def backward(dl_dr=None):
            if dl_dr is None:
                if len(result_data.shape) == 0:
                    dl_dr = Tensor(1)
                elif len(result_data.shape) == 1 and result_data.shape[0] == 1:
                    dl_dr = Tensor([1])
                else:
                    raise GradError(
                        "If tensor is not scalar then backwards must be called with a gradient of the same shape."
                    )
            if dl_dr.data.shape != result_data.shape:
                raise GradError(
                    "dL_dthis isn't of the right shape. It needs to match the shape of the tensor we're calling backward on."
                )
            if self.requires_grad:
                # The elementwise product with ones is to account for a `dim` argument having collapsed the matrix along a given dimension.
                # We just broadcast the dl_dr appropriately to fit the shape of the original matrix.
                if keepdim:
                    # Then we don't need to change dl_dr because backward() will have expected it to match the shape of the result.
                    self_grad = dl_dr.data * torch.ones_like(
                        self.data, requires_grad=False
                    )
                else:
                    # Then we need to unsqueeze dl_dr because dl_dr would have lost a dimension in order to match the shape of the result.
                    self_grad = dl_dr.data.unsqueeze(dim=dim) * torch.ones_like(
                        self.data, requires_grad=False
                    )
                if self.is_leaf:
                    if self.grad is not None:
                        self.grad = self.grad + self_grad
                    else:
                        self.grad = self_grad
                else:
                    self.backward(self_grad)

        result.backward = backward

        return result

    def __matmul__(self, other):
        result_data = self.data @ other.data
        result = Tensor(
            result_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            is_leaf=False,
        )

        def backward(dl_dr=None):
            if dl_dr is None:
                if len(result_data.shape) == 0:
                    dl_dr = Tensor(1)
                elif len(result_data.shape) == 1 and result_data.shape[0] == 1:
                    dl_dr = Tensor([1])
                else:
                    raise GradError(
                        "If tensor is not scalar then backwards must be called with a gradient of the same shape."
                    )
            if dl_dr.data.shape != result_data.shape:
                raise GradError(
                    "dL_dthis isn't of the right shape. It needs to match the shape of the tensor we're calling backward on."
                )

            if self.requires_grad:
                self_grad = dl_dr.data @ other.data.T
                if self.is_leaf:
                    if self.grad is not None:
                        self.grad = self.grad + self_grad
                    else:
                        self.grad = self_grad
                else:
                    self.backward(self_grad)
            if other.requires_grad:
                other_grad = self.data.T @ dl_dr.data
                if other.is_leaf:
                    if other.grad is not None:
                        other.grad = other.grad + other_grad
                    else:
                        other.grad = other_grad
                else:
                    other.backward(other_grad)

        result.backward = backward

        return result
