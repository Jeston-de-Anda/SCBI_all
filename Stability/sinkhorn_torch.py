#!/usr/bin/env python
"""
Sinkhorn Implementations in Pytorch

J. Wang
Created on 2020-1-8
"""

import torch
import numpy as np




def normalize(mat, axis_sum, axis=1):
    """
    A general normalizer of matrices.

    Parameters
    ----------
    mat     : pytorch 2-tensor of size (n,m) (passed by reference, mutable)
    axis_sum: marginal sum, pytorch 1-tensor, size matches mat and axis (mutable)
    axis    : flag of normalization direction, 1 for row-, 0 for column-

    P.S. normalize() will change both mat and axis_sum, please make copies first,
         such as, call normalize(mat.clone(), axis_sum.clone(), axis)
    
    Return
    ------
    normalized matrix in place (mat).
    """
    div = axis_sum / torch.sum(mat, dim=axis)
    mat *= (div.view([-1,1]) if axis else div)
    # tested, this kind of expression is better than `div.view([(1,-1),(-1,1)][axis])`
    # also `if axis` is faster than `if axis==0`
    return mat

def row_normalize(mat, row_sum):
    '''
    Row-normalization
    See `normalize()`
    '''
    div = row_sum / torch.sum(mat, dim=1)
    mat *= div.view([-1,1])
    return mat

def col_normalize(mat, col_sum):
    '''
    Column-normalization
    See `normalize()`
    '''
    div = col_sum / torch.sum(mat, dim=0)
    mat *= div
    return mat
    

def sinkhorn_torch(mat,
                   row_sum=None,
                   col_sum=None,
                   epsilon=1e-7,
                   max_iter=10000):
    '''
    Sinkhorn scaling

    Parameters
    ----------
    mat     : muted torch 2-tensor of shape(n,m)
    row_sum : immuted torch 1-tensor of size n
    col_sum : immuted torch 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''
    n, m = mat.shape
    row_sum = row_sum if row_sum is not None else torch.ones(n, dtype=torch.float64,
                                                             device=mat.device)
    col_sum = col_sum if col_sum is not None else torch.ones(m, dtype=torch.float64,
                                                             device=mat.device)

    diff = torch.ones(m, dtype=torch.float64, device=mat.device)
    max_iter //= 10
    
    while torch.sum(torch.abs(diff)) >= epsilon and max_iter:
        for i in range(10):
            row_normalize(col_normalize(mat, col_sum), row_sum)
        diff = col_sum - torch.sum(mat, axis=0)
        max_iter -= 1

    return mat
