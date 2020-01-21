#!/usr/bin/env python
"""
Equi-RoC Matrix Pertubations

In 3 by 3 case, we choose true hypo = 0 and sort the other 2 columns accoording to their
normalized-KL value. Then we perturb the irrelevant column, the last column, with keeping
the normalized-KL value the same. Then we try to investigate the influence of such changes.
"""

import pickle
import numpy as np
import torch
# import stability_testers as st
import sinkhorn_torch as sk

TRANS_TORCH = torch.tensor([[-1, -1, 2],
                            [-np.sqrt(3), np.sqrt(3), 0]], dtype=torch.float64)/np.sqrt(6)

UPPERBOUND = lambda theta: 1./np.sqrt(6)/np.sin(5/6*np.pi-theta)


def adjust_matrix(matrix):
    """
    Sorting matrix cols.

    matrix: can be a numpy 2d-array or pytorch 2d-Tensor

    Return
    ------
    adjusted pytorch 2d-tensor
    """
    if isinstance(matrix, np.ndarray):
        tmp = torch.from_numpy(matrix).clone() # ?
    else:
        tmp = matrix.clone()

    tmp /= tmp[:, 0].view([-1, 1])
    tmp = sk.col_normalize(tmp, torch.ones(3, dtype=torch.float64))

    if torch.sum(torch.log(tmp[:, 1])) > torch.sum(torch.log(tmp[:, 2])):
        # return torch.from_numpy(matrix)
        return 2

    return 1
    # ref = matrix[:, 1].copy()
    # matrix[:, 1] = matrix[:, 2]
    # matrix[:, 2] = ref

    # return torch.from_numpy(matrix)


def generate_single(matrix, index, density=120):
    """
    For a single matrix, generate equi-RoC variations

    Further: structure has a $S_3$ symmetry, we may just calculate 1/6 of it.

    Parameters
    ----------
    matrix: a pytorch 2d-tensor, target matrix which is adjusted
    index: target column, can only be 1 or 2
    density:
    """
    n_row, n_col = matrix.shape
    tmp = matrix.clone()
    col = tmp[:, 0].view([-1, 1]).clone()
    # print(col)
    tmp /= col
    tmp = sk.col_normalize(tmp, torch.ones(n_col, dtype=torch.float64))
    origin = torch.ones(n_row, dtype=torch.float64)/n_row
    destination = torch.sum(torch.log(tmp[:, index]))

    theta_array = np.linspace(0, 2*np.pi, density, endpoint=False)

    result = []
    for theta in theta_array:
        inner = tmp.clone()
        # print(col)
        weight = newton(0.5, loss, theta, origin, destination, lower=0, upper=UPPERBOUND(0.5))
        inner[:, index] = recover_coords(weight, theta, origin, destination)
        inner *= col
        result += [sk.col_normalize(inner, torch.ones(n_col, dtype=torch.float64)), ]
    return result


def recover_coords(ratio, *args):
    """
    calculate correct coordinate in simplex
    """

    theta, origin = args[0], args[1]
    return origin + torch.tensor([ratio, ], dtype=torch.float64) * \
        torch.matmul(torch.tensor([np.cos(theta), np.sin(theta)],
                                  dtype=torch.float64), TRANS_TORCH)


def newton(start, loss_fn, *args, lower=0, upper=None, epsilon=1e-9):
    """
    Newton's Method!
    """
    theta, origin, destination = args[0], args[1], args[2]

    if upper is None:
        upper = 1

    start = lower

    while True:
        if loss_fn(start, theta, origin, destination) > 0:
            start = (upper+start)/2
        else:
            start = (lower+start)/2

        # print("START", start)
        x_cur = start
        x_prev = -1
        try:
            while np.abs(x_cur-x_prev) >= epsilon:

                # print(x)
                x_prev = x_cur
                x_cur = newton_single(x_cur, loss_fn, theta, origin, destination)
                # print(x, x-x_prev, np.abs(x-x_prev)>=epsilon)
            if np.isnan(x_cur):
                continue
            return x_cur
        except ZeroDivisionError:
            print(start, x_cur)


def loss(ratio, *args):
    """
    Loss function, sum of log of col given by parameter ratio
    """
    theta, origin, destination = args
    tmp = origin + ratio * torch.matmul(torch.tensor([np.cos(theta), np.sin(theta)],
                                                     dtype=torch.float64), TRANS_TORCH)
    loss_value = torch.sum(torch.log(tmp)) - destination
    return loss_value


def newton_single(ratio, loss_fn, *args):
    '''
    Parameters
    ----------
    x   : a number or a list
    loss: function accepting temp t

    Returns
    -------
    A new value? (Pytorch这个要怎么用啊)
    '''
    tmp = torch.tensor([ratio], requires_grad=True)
    l_value = loss_fn(tmp, *args)
    l_value.backward()
    return (tmp - l_value/tmp.grad.data)[0].item()

def run(density=90, filename="Dim3_setup.log", outname="Equi_RoC.log"):
    """
    Run the work on Dim3_setup.log
    """

    with open(filename, "rb") as file_ptr:
        data = pickle.load(file_ptr)

    mats = data["mats"]
    priors = data["priors"]

    mat_vars = []
    for mat in mats:
        index = adjust_matrix(mat.numpy())
        mat_vars += [(mat, generate_single(mat, index, density=density)),]

    with open(outname, "wb") as f_ptr:
        pickle.dump({"mats": mat_vars,
                     "priors": priors}, f_ptr)


if __name__ == '__main__':
    M = np.random.dirichlet([1, 1, 1], 3).T
    adjust_matrix(M)
    generate_single(M, 2)

    run()
