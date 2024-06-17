import torch

from tensor import Tensor


def test_add_backwards_lhs():
    x_list = [[1, 2], [3, 5]]
    y_list = [[1, 2], [3, 5]]
    dl_dz_list = [[1, 2], [3, 4]]
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
    x_list = [[1, 2], [3, 5]]
    y_list = [[1, 2], [3, 5]]
    dl_dz_list = [[1, 2], [3, 4]]
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
    x_list = [[1, 2], [3, 5]]
    y_list = [[1, 2], [3, 5]]
    dl_dz_list = [[1, 2], [3, 4]]
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
