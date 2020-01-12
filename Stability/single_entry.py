'''
This is python file generating a single entry perturbation
on a given matrix.
'''

import numpy as np

def perturb_single_entry(matrices, i, j, epsilon=0.01, seed=None):
    '''
    Sample perturbation on a single entry

    Parameters
    ----------
    matrices: array of matrices
    i       : i-th row
    j       : j-th column
    epsilon : scale of perturbation
    seed    : random seed

    Return
    ------
    An array of perturbed matrices
    '''
    if seed is not None:
        np.random.seed(seed)
    mat = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    l = mat.shape[0]
    # factor = np.ones_like(matrices)
    mat[:, i, j] *= (1 + epsilon * (2 * np.random.rand(l) - 1))
    return mat



if __name__ == "__main__":
    print(perturb_single_entry(np.array([[1, 2], [3, 4]]), 0, 1))
