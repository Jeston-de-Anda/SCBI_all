import multiprocessing as mp
import numpy as np
import pickle
import datetime

import head_to_head

import sys

def write_stdout():
    sys.stdout = open(str(datetime.datetime.today()) + ".log", "w")

def write_stderr():
    sys.stderr = open(str(datetime.datetime.today()) + ".err", "w")

# cofigurations
RANK = 10
SAMPLE_SIZE = 1000000000
SINGLE_SAMPLE_SIZE = 2000
NUMBER_OF_COLUMNS = 2
FOR_LOOP_DEPTH = 500
# functions

def refresh_config():
    global RANK, SAMPLE_SIZE, SINGLE_SAMPLE_SIZE, NUMBER_OF_COLUMNS
    RANK = 10
    SAMPLE_SIZE = 1000000000
    SINGLE_SAMPLE_SIZE = 2000
    NUMBER_OF_COLUMNS = 2



def sample_mat_default(row_size=RANK, col_size=RANK, size=SINGLE_SAMPLE_SIZE):
    '''
    default sampling function.
    return an array of matrices of shape row_size*col_size,
    '''
    return np.random.dirichlet([1, ]*row_size,
                               col_size * size).reshape(-1, col_size, row_size).transpose(0, 2, 1)


def roc_bi(matrices):
    '''
    Calculate KL-divergence given an array of matrix. This is
    naturally the RoC of BI.

    Parameters
    ----------
    matrices: an array of matrices, can be only one matrix. Must be column-normalized!

    Return
    ------
    l*m numpy array of RoC_BI, denoting the number of matrices with each hypothesis chosen
    as the correct one.
    '''
    mat = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    l, n, m = mat.shape
    log = np.log(mat)   # will be used twice, calculate independently.
    # next we calculate for each matrix and hypothesis, the KL-divergence,
    # res in terms of (l,m,m) array, l matrices, m true hypotheses, m ref hypotheses
    res = np.sum(mat.reshape(l, n, 1, m) * (log.reshape(l, n, 1, m) - log.reshape(l, n, m, 1)),
                 axis=1)
    # next find the minimal nonzero elements among the 'other hypotheses'
    return np.min(res + np.diag(np.ones(m)*np.inf).reshape(1, m, m), axis=1)


def roc_scbi(matrices):
    '''
    Calculate the rate of convergence of SCBI.

    Parameters
    ----------
    same as roc_bi()

    Return
    ------
    same format as roc_bi()
    '''
    mat = matrices.reshape(-1, matrices.shape[-2], matrices.shape[-1])
    l, n, m = mat.shape
    logm = np.log(mat)
    # print(logm)
    # 4 indices: (mat, row-index, other-col-index, true-col-index), same as in BI.
    quotient = np.sum(mat.reshape(l, n, m, 1)/mat.reshape(l, n, 1, m), axis=1)
    logq = np.log(quotient)  + np.diag(np.ones(m)*np.inf).reshape(1, m, m)
    # print(logq)
    logs = np.mean(logm, axis=1)
    other = logs.reshape(l, 1, m) - logs.reshape(l, m, 1) + logq
    # print(other)
    return np.min(other, axis=1) - np.log(n)


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
    diff = np.mean(roc_scbi(matrices), axis=1) - np.mean(roc_bi(matrices), axis=1)
    return np.array([np.mean(diff), np.mean((diff >= 0))])

def single_round(args):
    '''
    Single round in multiprocessing
    '''
    global SINGLE_SAMPLE_SIZE, FOR_LOOP_DEPTH

    seed, n, m = args
    np.random.seed(seed)
    result = np.zeros([FOR_LOOP_DEPTH, 2])
    for i in range(FOR_LOOP_DEPTH):
        matrices = np.random.dirichlet([1,]*n,
                                       m * SINGLE_SAMPLE_SIZE).reshape(-1, m, n).transpose(0, 2, 1)
        result[i,:] = comparison(matrices)

    return np.mean(result, axis=0)

def multi_round(n, m):
    '''
    Multi-Processing
    '''
    global SAMPLE_SIZE

    number_of_process = int(SAMPLE_SIZE / SINGLE_SAMPLE_SIZE / FOR_LOOP_DEPTH)
    seeds = np.random.randint(int(2**32), size=number_of_process)
    args = [(x, n, m) for x in seeds]

    pool = mp.Pool()
    print("\n", n, "rows", m, "colums")
    result = np.array(pool.map(single_round, args))
    pool.close()
    return np.mean(result, axis=0)

def run10(start, size, sample=None, single = None):
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
        with open("fix_comparison_"+str(size)+"_by_"+str(i)+".log", "wb") as fp:
            pickle.dump(r[-1], fp)

    refresh_config()
    return r

def run_full(start, size, sample=None, single=None):
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
            pickle.dump(r[-1], fp)

    refresh_config()
    return r

if __name__ == "__main__":
    pass
