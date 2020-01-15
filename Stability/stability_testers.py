#!/usr/bin/env python
'''
This is a file to test stability.
Pytorch version, allow to calculate using CPU/GPU/Multi-GPU

J. Wang.
Created on 2020-1-6
'''
import numpy as np
import torch
import torch.multiprocessing as mp
import sinkhorn_torch as sk
import datetime


from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class StabilityTester:
    """
    This is the tester class of SCBI stability.
    """
    def __init__(self, device):
        '''
        Stability Tester
        '''

        if device.lower() == "cpu":
            self.device = "cpu"
        elif device.lower()[:4] == "cuda":
            if device[4:] == "":
                self.device = "cuda"
            elif device[5:].isnumeric():
                if int(device[5:]) < torch.cuda.device_count():
                    self.device = "cuda:" + device[5:]
        else:
            self.device = "cpu"
            # raise Exception("Unavailable device") # another choice

        # self.block_size=0

        self.sk_epsilon = 1e-7
        self.bi_epsilon = 1e-5
        self.max_depth = 1000
        self.cpu = torch.device("cpu")


    def set_mat_teach(self, mat_teach):
        "Set teaching matrix"
        self.mat_teach = mat_teach.cpu()

    def set_mat_learn(self, mat_learn):
        "Set learning matrix"
        self.mat_learn = mat_learn.cpu()

    def set_prior_teach(self, prior_teach):
        "Set teaching priorrix"
        self.prior_teach = prior_teach.cpu()

    def set_prior_learn(self, prior_learn):
        "Set learning priorrix"
        self.prior_learn = prior_learn.cpu()

    def get_mat_teach(self):
        "Get teaching matrix"
        return self.mat_teach

    def get_mat_learn(self):
        "Get learning matrix"
        return self.mat_learn

    def get_prior_teach(self):
        "Get teaching priorrix"
        return self.prior_teach

    def get_prior_learn(self):
        "Get learning priorrix"
        return self.prior_learn

    def set_correct_hypo(self, hypo):
        "Set correct hypothesis to teach"
        self.correct_hypo = hypo

    def get_correct_hypo(self):
        "Get correct hypothesis to teach"
        return self.correct_hypo

    def single_inference(self):
        '''
        One Episode Inference
        '''
        theta_teach = self.prior_teach.clone()
        theta_learn = self.prior_learn.clone()
        n, m = self.mat_teach.shape
        count = 0

        while theta_teach[self.correct_hypo] < 1. - self.bi_epsilon and \
          count < self.max_depth:

            count += 1
            theta_teach *= n
            theta_learn *= n

            mat_teach = self.mat_teach.clone()
            mat_learn = self.mat_learn.clone()

            sk.sinkhorn_torch(mat_teach,
                              col_sum=theta_teach,
                              epsilon=self.sk_epsilon)

            sk.sinkhorn_torch(mat_learn,
                              col_sum=theta_learn,
                              epsilon=self.sk_epsilon)

            prob = mat_teach[:, self.correct_hypo].clone()
            prob /= torch.sum(prob)
            d = torch.distributions.categorical.Categorical(prob).sample()

            theta_teach = mat_teach[d, :]
            theta_learn = mat_learn[d, :]
            # print(count, d, theta_teach[self.correct_hypo], theta_learn[self.correct_hypo])

        # return theta_teach[self.correct_hypo].item(), theta_learn
        return torch.cat([theta_teach, theta_learn])


    def single(self, seed):
        """
        integrated single-process handle

        Parameters
        ----------
        seed: the seed of pytorch random for each process.

        Return
        ------
        a vector of 2m length denoting the average posterior for
        both teacher and learner side.
        """

        self.mat_teach = self.mat_teach.to(self.device)
        self.prior_teach = self.prior_teach.to(self.device)
        self.prior_learn = self.prior_learn.to(self.device)
        self.mat_learn = self.mat_learn.to(self.device)

        torch.manual_seed(seed)
        n, m = self.mat_teach.shape
        ret = torch.zeros([self.block_size, m * 2],
                          dtype=torch.float64,
                          device=self.mat_teach.device)
        for i in range(self.block_size):
            ret[i] = self.single_inference().clone()

        return torch.mean(ret, dim=0).to(self.cpu)


    def inference_fixed_initial(self, repeats, block_size=100, pool_size=10):
        '''
        Inference behaviour on a single initial condition

        Parameters
        ----------
        repeats   : the number of processes (blocks)
        block_size: number of episodes in each block (process)

        Return
        ------
        torch 2d-tensor:
        first row is the expectation of teacher's posterior
                  (within tolerance, non-fixed step sizes)
        second row is the expectation of learner's posterior
                   this is the supposed final distribution on
                   the vertices, if first row is close enough
                   to delta-distribution.
        '''

        print("Inference Start:", datetime.datetime.today())

        n, m = self.mat_teach.shape
        self.block_size = block_size
        # maybe it worths to do it also on GPU?


        seeds = list(np.random.randint(0, int(2**31-1), [repeats,]))
        pool = mp.Pool(pool_size)

        # result = torch.cat(pool.map(self.single, seeds)).view([-1, 2 * m])
        inf_result = pool.map(self.single, seeds)
        pool.close()
        inf_result = torch.cat(inf_result).view([-1, 2 * m])
        # print(inf_result)

        self.mat_teach = self.mat_teach.to(self.cpu)
        self.prior_teach = self.prior_teach.to(self.cpu)
        self.prior_learn = self.prior_learn.to(self.cpu)
        self.mat_learn = self.mat_learn.to(self.cpu)


        print("Inference End:", datetime.datetime.today())
        return torch.mean(inf_result, dim=0).view([2,m])


if __name__ == '__main__':

    tester = StabilityTester("cuda")

    M=torch.tensor([[.1, .2], [.3, .4]], device=torch.device("cuda"))
    tester.set_correct_hypo(0)
    tester.set_mat_learn(M)
    tester.set_mat_teach(M)
    tester.set_prior_learn(torch.tensor([.5, .5], device=torch.device("cuda")))
    tester.set_prior_teach(torch.tensor([.6, .4], device=torch.device("cuda")))

    result = tester.inference_fixed_initial(10, 1)
    print(result)
