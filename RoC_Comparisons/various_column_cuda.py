#!/usr/bin/env python
"""
Use Pytorch to calculate multi-column comparison
of the RoC between BI and SCBI



J. Wang
"""
import torch.multiprocessing as mp
import numpy as np
import torch
import pickle
import datetime

# import head_to_head

import sys

def write_stdout():
    sys.stdout = open(str(datetime.datetime.today()) + ".log", "w")

def write_stderr():
    sys.stderr = open(str(datetime.datetime.today()) + ".err", "w")

# cofigurations
RANK = 10
SAMPLE_SIZE = 100000000
SINGLE_SAMPLE_SIZE = 200
NUMBER_OF_COLUMNS = 2
FOR_LOOP_DEPTH = 500
DEVICE = torch.device("cuda")
PROCESS_WIDTH = 10

# functions


def set_device(device_name):
    global DEVICE, PROCESS_WIDTH
    DEVICE = torch.device(device_name)
    if device_name == "cpu":
        PROCESS_WIDTH=10
    else:
        PROCESS_WIDTH=4

def refresh_config():
    global RANK, SAMPLE_SIZE, SINGLE_SAMPLE_SIZE, NUMBER_OF_COLUMNS
    RANK = 10
    SAMPLE_SIZE = 1000000000
    SINGLE_SAMPLE_SIZE = 200
    NUMBER_OF_COLUMNS = 2

def sample_mat_default(row_size=RANK, col_size=RANK, size=SINGLE_SAMPLE_SIZE, device=DEVICE):
    '''
    default sampling function.
    return an array of matrices of shape row_size*col_size,
    '''
    return torch._sample_dirichlet(torch.tensor([[1., ] * row_size, ] * (col_size * size),
                                                dtype=torch.float64,
                                                device=device)).reshape(-1, col_size, row_size).permute(0, 2, 1)



def roc_bi(mat):
    '''
    Calculate KL-divergence given an array of matrix. This is
    naturally the RoC of BI.

    Parameters
    ----------
    mat: an array of matrices, can be only one matrix. Must be column-normalized!

    Return
    ------
    l*m numpy array of RoC_BI, denoting the number of matrices with each hypothesis chosen
    as the correct one.
    '''
    mat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
    l, n, m = mat.shape
    log = torch.log(mat)   # will be used twice, calculate independently.
    # next we calculate for each matrix and hypothesis, the KL-divergence,
    # res in terms of (l,m,m) array, l matrices, m true hypotheses, m ref hypotheses
    logdiff = log.reshape(l, n, 1, m) - log.reshape(l, n, m, 1)
    logdiff *= mat.reshape(l, n, 1, m)

    res = torch.sum(logdiff, dim=1)

    del log, logdiff
    torch.cuda.empty_cache()
    # next find the minimal nonzero elements among the 'other hypotheses'
    return torch.min(res + \
                     torch.diag(torch.ones(m,
                                           device=res.device,
                                           dtype=torch.float64)*float("Inf")).reshape(1, m, m),
                     dim=1).values


def roc_scbi(mat):
    '''
    Calculate the rate of convergence of SCBI.

    Parameters
    ----------
    same as roc_bi()

    Return
    ------
    same format as roc_bi()
    '''
    mat = mat.reshape(-1, mat.shape[-2], mat.shape[-1])
    l, n, m = mat.shape
    logm = torch.log(mat)
    # print(logm)
    # 4 indices: (mat, row-index, other-col-index, true-col-index), same as in BI.
    bigmat = mat.reshape(l, n, m, 1)/mat.reshape(l, n, 1, m)
    quotient = torch.sum(bigmat, dim=1)

    logq = torch.log(quotient) + \
        torch.diag(torch.ones(m,
                              dtype=torch.float64,
                              device=logm.device)*float("Inf")).reshape(1, m, m)
    torch.cuda.empty_cache()

    logs = torch.mean(logm, dim=1)
    logq += logs.reshape(l, 1, m)
    logq -= logs.reshape(l, m, 1)
    result = torch.min(logq, axis=1).values - np.log(n)

    del quotient, bigmat
    del logm, logq
    torch.cuda.empty_cache()

    return result


def comparison(matrices):
    '''
    Comparison of BI and SCBI

    Parameters
    ----------
    matrices: same as roc_bi()

    Return
    ------
    np.array([avg, prob])
    avg : average of roc_bi - roc_scbi
    prob: frequency of roc_bi <= roc_scbi
    '''
    diff = torch.mean(roc_scbi(matrices), dim=1)
    diff -= torch.mean(roc_bi(matrices), dim=1)
    # torch.cuda.empty_cache()
    # return [torch.mean(diff).item(), torch.mean((diff >= 0).double()).item()]
    return torch.mean(diff).item(), torch.sum((diff >= 0).long()).item()


def single_round(args):
    '''
    Single round in multiprocessing
    '''
    global SINGLE_SAMPLE_SIZE, FOR_LOOP_DEPTH, DEVICE

    seed, n, m = args
    torch.manual_seed(seed)
    # result = np.zeros([FOR_LOOP_DEPTH, 2])
    # result = torch.zeros([FOR_LOOP_DEPTH, 2],
    #                      dtype=torch.float64,
    #                      device=torch.device("cpu"))
    mean = np.zeros(FOR_LOOP_DEPTH, dtype=np.float64)
    freq = np.zeros(FOR_LOOP_DEPTH, dtype=np.int64)
    for i in range(FOR_LOOP_DEPTH):
        matrices = sample_mat_default(n, m, SINGLE_SAMPLE_SIZE, DEVICE)
        # result[i,:] = comparison(matrices)
        mean[i], freq[i] = comparison(matrices)
    # return torch.mean(result, dim=0).numpy()
    return np.mean(mean), np.sum(freq)
    # return np.sum(np.array(result), axis=0)

def multi_round(n, m):
    '''
    Multi-Processing
    '''
    global SAMPLE_SIZE, PROCESS_WIDTH

    number_of_process = int(SAMPLE_SIZE / SINGLE_SAMPLE_SIZE / FOR_LOOP_DEPTH)
    seeds = np.random.randint(int(2**31-1), size=number_of_process)
    args = [(x, n, m) for x in seeds]

    pool = mp.Pool(PROCESS_WIDTH)
    print("\n", n, "rows", m, "colums")
    result = pool.map(single_round, args)
    multi_mean = np.array([x[0] for x in result])
    multi_freq = np.array([x[1] for x in result])
    pool.close()
    return np.mean(multi_mean), np.sum(multi_freq)

def run_fix_row(start, size, sample=None, single=None):
    '''
    fix row number, vary column number
    '''
    global SAMPLE_SIZE, SINGLE_SAMPLE_SIZE
    if sample is not None:
        SAMPLE_SIZE = sample
        SINGLE_SAMPLE_SIZE = single
    r = []
    print(datetime.datetime.today())
    for i in range(start, size + 1):
        r += [multi_round(size, i), ]
        print("\n", r[-1])
        print(datetime.datetime.today())
        with open("fix_row_"+str(size)+"_by_"+str(i)+".log",
                  "wb") as fp:
            pickle.dump([r[-1], SAMPLE_SIZE], fp)

    refresh_config()
    return r

def run_square(start, size, sample=None, single=None):
    '''
    square matrix test
    '''
    global SAMPLE_SIZE, SINGLE_SAMPLE_SIZE
    if sample is not None:
        SAMPLE_SIZE = sample
        SINGLE_SAMPLE_SIZE = single

    r = []
    print(datetime.datetime.today())
    for i in range(start, size + 1):
        r += [multi_round(i, i)]
        print(r[-1])
        print(datetime.datetime.today())
        with open("square_"+str(i)+".log", "wb") as fp:
            pickle.dump([r[-1],SAMPLE_SIZE], fp)

    refresh_config()
    return r

if __name__ == "__main__":
    pass
