import multiprocessing as mp
import numpy as np
from scipy import integrate
import itertools
import pickle
import datetime

def f(x,y):
    return np.sum((x-y) * np.log(x/y),axis=1)
    
def g(x,y):
    return np.log(np.sum(x/y,axis=1) * np.sum(y/x,axis=1)) - 2 * np.log(x.shape[1])
    
    

def single_process(x):
    pack = 100
    seed, rank, N = x 
    # assume N is multiple of pack
    np.random.seed(seed)
    M = N//pack
    mean = np.zeros(pack)
    prob = np.zeros(pack)
    for i in range(pack):
        x = np.random.dirichlet([1,] * rank, M)
        y = np.random.dirichlet([1,] * rank, M)
        diff = g(x,y) - f(x,y)
        mean[i] = np.mean(diff)
        prob[i] = np.mean(diff >= 0)
    
    # logs = np.log(np.sum(np.random.dirichlet([1,] * rank, N)/np.random.dirichlet([1,] * rank, N),axis=1))
    return np.mean(mean), np.mean(prob)

def multi_process(rank=2, single=10000000, rounds=100):
    pool = mp.Pool()
    seeds = np.random.randint(0,2**32,size=rounds)
    raw_result = pool.map(single_process, 
                            zip(seeds,
                                itertools.repeat(rank, rounds), 
                                itertools.repeat(single, rounds)))
    pool.close()
    result = np.array(raw_result).T
    return np.mean(result[0]), np.mean(result[1])

def run(process=1000,
        lower_bound = 50,
        upper_bound =51,
        ):
    for i in range(lower_bound, upper_bound):
        res = multi_process(i, 10000000, process)
        with open("rank_"+str(i)+".log", "wb") as fp:
            pickle.dump(res, fp)
        print("rank", str(i), ":", res)
        print(datetime.datetime.today())

if __name__ == "__main__":
    run()