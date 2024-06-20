"""Tensor operations and autograd"""

import tomllib

import torch


class GradError(Exception):
    pass


def dl_dr_check(dl_dr, tensor):
    """Check if the dl_dr provided to a call to backward() matches the data in question.

    The main cases are:
    - grad is None, which happens when someone calls backward() without supplying an upstream gradient. This only makes
      sense if the tensor is a scalar, in which case we just make the upstream grad 1.
    - Otherwise the grad needs to match the shape of the tensor
    """

    if dl_dr is None:
        if len(tensor.data.shape) == 0:
            dl_dr = Tensor(1)
        elif len(tensor.data.shape) == 1 and tensor.data.shape[0] == 1:
            dl_dr = Tensor([1])
        else:
            raise GradError(
                "If tensor is not scalar then backwards must be called with a gradient of the same shape."
            )
    if dl_dr.data.shape != tensor.data.shape:
        raise GradError(
            "dL_dthis isn't of the right shape. It needs to match the shape of the tensor we're calling backward on."
        )
    return dl_dr


def update_gradients_and_propagate(tensor, grad):
    """Given a tensor, and the gradient of the loss with respect to that
    tensor, update the tensors gradient property or backpropagate further down
    the graph
    """
    if tensor.is_leaf:
        if tensor.grad is not None:
            tensor.grad = tensor.grad + grad
        else:
            tensor.grad = grad
    else:
        tensor.backward(grad)


class Tensor:
    """Tensor class implementation for the purposes of autograd. For the raw
    forward operations, we just defer to torch.Tensor implementation. But this
    implementation provides backward() operations which can be recursively
    called on dependent tensors to do backward propagation of gradients all the
    way to the leaf nodes.

    For the backward() functions, the general idea is that for binary operations
    we have `r = self + other`. `r` is the result. `backward` is given dl/dr (the
    derivative of the loss with respect to the result r). The goal is to:
        1. compute dl/dother and dl/dself.
        2. If self or other are not leaf nodes, then we call their backward() with dl/dother and dl/dself
    """

    def __init__(
        self, data=[], dtype=None, device="cpu", requires_grad=False, is_leaf=True
    ):
        if isinstance(data, torch.Tensor):
            self.data = data.clone().detach()
        else:
            if dtype is not None:
                self.data = torch.tensor(
                    data, dtype=torch.float32, device=device, requires_grad=False
                )
            else:
                self.data = torch.tensor(
                    data, dtype=dtype, device=device, requires_grad=False
                )
        self.is_leaf = is_leaf
        self.requires_grad = requires_grad
        self.grad = None
        self.backward = None

    @property
    def shape(self):
        return self.data.shape

    def __iter__(self):
        for item in self.data:
            yield Tensor(item)

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def __setitem__(self, index, value):
        if isinstance(value, Tensor):
            self.data[index] = value.data
        else:
            self.data[index] = value

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

            In the case of the sums, we have:
            - dr/dself = a tensor of ones: As each element of dself changes, the corresponding element of r changes proportionally
            - dl/dself = dl/dr * dr/dself: Each element of self can only affect the loss via the corresponding element of r, so we just take the element-wise
            product of dl/dr with the ones tensor, resulting in dl/dr tensor again.
            """
            dl_dr = dl_dr_check(dl_dr, result)
            if self.requires_grad:
                self_grad = dl_dr.data
                update_gradients_and_propagate(self, self_grad)
            if other.requires_grad:
                other_grad = dl_dr.data
                update_gradients_and_propagate(other, other_grad)

        result.backward = backward

        return result

    def sum(self, dim=None, keepdim=False):
        result_data = self.data.sum(dim=dim, keepdim=keepdim)
        result = Tensor(result_data, requires_grad=self.requires_grad, is_leaf=False)

        def backward(dl_dr=None):
            dl_dr = dl_dr_check(dl_dr, result)
            if self.requires_grad:
                # The elementwise product with ones is to account for a `dim` argument having collapsed the matrix along a given dimension.
                # We just broadcast the dl_dr appropriately to fit the shape of the original matrix.
                if keepdim:
                    # Then we don't need to change dl_dr because backward() will have expected it to match the shape of the result.
                    self_grad = dl_dr.data * torch.ones_like(
                        self.data, requires_grad=False
                    )
                elif dim:
                    # Then we need to unsqueeze dl_dr because dl_dr would have lost a dimension in order to match the shape of the result.
                    self_grad = dl_dr.data.unsqueeze(dim=dim) * torch.ones_like(
                        self.data, requires_grad=False
                    )
                else:
                    # Then dl_dr needs to match the shape of result, so we just have
                    self_grad = dl_dr.data
                update_gradients_and_propagate(self, self_grad)

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
            dl_dr = dl_dr_check(dl_dr, result)
            if self.requires_grad:
                self_grad = dl_dr.data @ other.data.T
                update_gradients_and_propagate(self, self_grad)
            if other.requires_grad:
                other_grad = self.data.T @ dl_dr.data
                update_gradients_and_propagate(other, other_grad)

        result.backward = backward

        return result


def torchify(arg):
    if isinstance(arg, tuple):
        return tuple(torchify(a) for a in arg)
    elif isinstance(arg, Tensor):
        return arg.data
    else:
        return arg


def create_wrapper(torch_func):
    """Recall that our Tensor class is just a wrapper around torch.Tensor that
    uses it's own autograd and backwards() functions but defers to the torch
    implementation for the 'forward' operations. The idea here in
    `create_wrapper` is to create our own versions of torch functions that just
    rely on applying the torch function to the underlying data housed in our
    Tensor."""

    def f(*args, **kwargs):
        args = [torchify(arg) for arg in args]
        if "requires_grad" in kwargs:
            requires_grad = kwargs["requires_grad"]
            kwargs["requires_grad"] = False
        else:
            requires_grad = False
        data = torch_func(*args, **kwargs)
        dtype = data.dtype
        device = data.device
        return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    return f


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

if config["torch_autograd"]:
    tensor = torch.tensor
    randint = torch.randint
    zeros = torch.zeros
    tril = torch.tril
    ones = torch.ones
    cat = torch.cat
    arange = torch.arange
    argmax = torch.argmax
    topk = torch.topk
    multinomial = torch.multinomial
    zeros_like = torch.zeros_like
    sqrt = torch.sqrt
    erf = torch.erf
    exp = torch.exp
    Tensor = torch.Tensor
    empty = torch.empty
    einsum = torch.einsum
    randn = torch.randn
    gather = torch.gather
    ones_like = torch.ones_like
    bernoulli = torch.bernoulli
else:
    randint = create_wrapper(torch.randint)
    zeros = create_wrapper(torch.randint)
    tril = create_wrapper(torch.tril)
    ones = create_wrapper(torch.ones)
    cat = create_wrapper(torch.cat)
    arange = create_wrapper(torch.arange)
    argmax = create_wrapper(torch.argmax)
    topk = create_wrapper(torch.topk)
    multinomial = create_wrapper(torch.multinomial)
    zeros_like = create_wrapper(torch.zeros_like)
    sqrt = create_wrapper(torch.sqrt)
    erf = create_wrapper(torch.erf)
    exp = create_wrapper(torch.exp)
    empty = create_wrapper(torch.empty)
    einsum = create_wrapper(torch.einsum)
    randn = create_wrapper(torch.randn)
    gather = create_wrapper(torch.gather)
    ones_like = create_wrapper(torch.ones_like)
    bernoulli = create_wrapper(torch.bernoulli)
    tensor = Tensor

if __name__ == "__main__":
    x = randint(2, (4,))
    y = randint(2, (4,))
    z = cat((x, y))
    x = 1
