import torch

from tensor import Tensor, GradError

x_list = [[1, 2], [3, 5]]
y_list = [[1, 2], [3, 6]]
w_list = [[6, 7], [8, 9]]
dl_dz_list = [[1, 2], [3, 4]]


def test_add_backwards_lhs():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=False)
    z = x + y
    dl_dz = Tensor(dl_dz_list)
    z.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=False, dtype=torch.float32)
    t_z = t_x + t_y
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z.backward(t_dl_dz)

    assert (x.grad == t_x.grad).all()
    assert y.grad == t_y.grad
    assert z.grad == t_z.grad


def test_add_backwards_rhs():
    x = Tensor(x_list, requires_grad=False)
    y = Tensor(y_list, requires_grad=True)
    z = x + y
    dl_dz = Tensor(dl_dz_list)
    z.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=False, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x + t_y
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z.backward(t_dl_dz)

    assert x.grad == t_x.grad
    assert (y.grad == t_y.grad).all()
    assert z.grad == t_z.grad


def test_add_backwards_bhs():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=True)
    z = x + y
    dl_dz = Tensor(dl_dz_list)
    z.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x + t_y
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z.backward(t_dl_dz)

    assert (x.grad == t_x.grad).all()
    assert (y.grad == t_y.grad).all()
    assert z.grad == t_z.grad


def test_add_backwards_multinode():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=True)
    w = Tensor(w_list, requires_grad=True)
    z1 = x + y
    z2 = z1 + w
    dl_dz = Tensor(dl_dz_list)
    z2.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=True, dtype=torch.float32)
    t_w = torch.tensor(w_list, requires_grad=True, dtype=torch.float32)
    t_z1 = t_x + t_y
    t_z2 = t_z1 + t_w
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z2.backward(t_dl_dz)

    assert (x.grad == t_x.grad).all()
    assert (y.grad == t_y.grad).all()
    assert (w.grad == t_w.grad).all()
    assert z1.grad == t_z1.grad
    assert z2.grad == t_z2.grad


def test_add_backwards_noarg_scalar_shape0():
    x = Tensor(2, requires_grad=True)
    y = Tensor(3, requires_grad=True)
    z = x + y  # z is a scalar
    z.backward()  # no argument

    t_x = torch.tensor(2, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(3, requires_grad=True, dtype=torch.float32)
    t_z = t_x + t_y
    t_z.backward()  # no argument

    assert (x.grad == t_x.grad).all()
    assert (y.grad == t_y.grad).all()


def test_add_backwards_noarg_scalar_shape1():
    x = Tensor([2], requires_grad=True)
    y = Tensor([3], requires_grad=True)
    z = x + y  # z is a scalar
    z.backward()  # no argument

    t_x = torch.tensor([2], requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor([3], requires_grad=True, dtype=torch.float32)
    t_z = t_x + t_y
    t_z.backward()  # no argument

    assert (x.grad == t_x.grad).all()
    assert (y.grad == t_y.grad).all()


def test_add_backwards_noarg_nonscalar():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=True)
    z = x + y  # z is non-scalar
    try:
        z.backward()  # no argument, should raise an error
        assert False, "Expected RuntimeError"
    except GradError:
        pass


def test_sum_backwards():
    x = Tensor(x_list, requires_grad=True)
    z = x.sum()
    z.backward()  # no argument because z is a scalar

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x.sum()
    t_z.backward()  # no argument because t_z is a scalar

    assert (x.grad == t_x.grad).all()
    assert z.grad == t_z.grad


def test_sum_dim0_backwards():
    x = Tensor(x_list, requires_grad=True)
    z = x.sum(dim=0)
    z.backward(
        torch.arange(x.data.shape[1])
    )  # torch.ones_like(z.data))  # supply a tensor of ones with the same shape as z

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x.sum(dim=0)
    t_z.backward(
        torch.arange(x.data.shape[1])
    )  # supply a tensor of ones with the same shape as t_z

    assert (x.grad == t_x.grad).all()
    assert z.grad == t_z.grad


def test_sum_dim1_backwards():
    x = Tensor(x_list, requires_grad=True)
    z = x.sum(dim=1)
    z.backward(
        torch.arange(x.data.shape[0])
    )  # supply a tensor of ones with the same shape as z

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x.sum(dim=1)
    t_z.backward(
        torch.arange(x.data.shape[0])
    )  # supply a tensor of ones with the same shape as t_z

    assert (x.grad == t_x.grad).all()
    assert z.grad == t_z.grad


def test_sum_dim1_keepdim_backwards():
    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x.sum(dim=1, keepdim=True)
    t_z.backward(
        torch.arange(t_x.data.shape[0])[:, None]
    )  # supply a tensor of ones with the same shape as t_z

    x = Tensor(x_list, requires_grad=True)
    z = x.sum(dim=1, keepdim=True)
    z.backward(
        torch.arange(x.data.shape[0])[:, None]
    )  # supply a tensor of ones with the same shape as z

    assert (x.grad == t_x.grad).all()
    assert z.grad == t_z.grad


def test_matmul_backwards_lhs():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=False)
    z = x @ y
    dl_dz = Tensor(dl_dz_list)
    z.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=False, dtype=torch.float32)
    t_z = t_x @ t_y
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z.backward(t_dl_dz)

    assert (x.grad == t_x.grad).all()
    assert y.grad == t_y.grad
    assert z.grad == t_z.grad


def test_matmul_backwards_rhs():
    x = Tensor(x_list, requires_grad=False)
    y = Tensor(y_list, requires_grad=True)
    z = x @ y
    dl_dz = Tensor(dl_dz_list)
    z.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=False, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=True, dtype=torch.float32)
    t_z = t_x @ t_y
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z.backward(t_dl_dz)

    assert x.grad == t_x.grad
    assert (y.grad == t_y.grad).all()
    assert z.grad == t_z.grad


def test_matmul_backwards_multinode():
    x = Tensor(x_list, requires_grad=True)
    y = Tensor(y_list, requires_grad=True)
    w = Tensor(w_list, requires_grad=True)
    z1 = x @ y
    z2 = z1 @ w
    dl_dz = Tensor(dl_dz_list)
    z2.backward(dl_dz)

    t_x = torch.tensor(x_list, requires_grad=True, dtype=torch.float32)
    t_y = torch.tensor(y_list, requires_grad=True, dtype=torch.float32)
    t_w = torch.tensor(w_list, requires_grad=True, dtype=torch.float32)
    t_z1 = t_x @ t_y
    t_z2 = t_z1 @ t_w
    t_dl_dz = torch.tensor(dl_dz_list)
    t_z2.backward(t_dl_dz)

    assert (x.grad == t_x.grad).all()
    assert (y.grad == t_y.grad).all()
    assert (w.grad == t_w.grad).all()
    assert z1.grad == t_z1.grad
    assert z2.grad == t_z2.grad
