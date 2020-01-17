#!/usr/bin/env python
"""
Experiment of Changing Priors

J. Wang
2020-1-14
"""

import pickle
import datetime
import numpy as np
import torch
import stability_testers as st
import sinkhorn_torch as sk


class PriorVariation:
    """
    Changing priors.

    Basic requirements on paramaters:
    ---------------------------------
    dim    : can be only 3 or 4
    method : for dim 3, "hex" or "circ"
             for dim 4, ""

    correct hypothesis: always 0
    change matrix columns: second row with minimal scbi_roc?
    """

    # dim 3 cases: For some historical simplicity reasons,
    # we chose to use row vectors as coordinates, so transformation
    # matrices are written in the row-format
    x_unit_3 = np.array((2, -1, -1), dtype=np.float64) / np.sqrt(6)
    y_unit_3 = np.array((-1, 2, -1), dtype=np.float64) / np.sqrt(6)
    trans_simplex_3 = np.array([x_unit_3, y_unit_3])
    x_vis_3 = np.array((0, 1), dtype=np.float64)
    y_vis_3 = np.array((np.sqrt(3)/2, -1/2), dtype=np.float64)
    trans_visual_3 = np.array([x_vis_3, y_vis_3])


    # dim 4 cases:

    trans_simplex_4 = np.array([[3, -1, -1, -1],
                                [-1, 3, -1, -1],
                                [-1, -1, -1, 3]],
                               dtype=np.float64) / (np.sqrt(12))

    trans_visual_4 = np.array([[np.sqrt(8) / 3, 0, -1 / 3],
                               [-np.sqrt(2) / 3, np.sqrt(2 / 3), -1 / 3],
                               [0, 0, 1]], dtype=np.float64)

    trans_simplex = {3: trans_simplex_3,
                     4: trans_simplex_4}

    trans_visual = {3: trans_visual_3,
                    4: trans_visual_4}


    @staticmethod
    def angle_to_std(phi, theta):
        """
        Latitude and Longitude to standard coordinates

        Parameters
        ----------
        phi, theta: must be numpy 1d-arrays.
        """
        return np.array([np.sin(phi)*np.cos(theta),
                         np.sin(phi)*np.sin(theta),
                         np.cos(phi)], dtype=np.float64).T


    @staticmethod
    def gen_hex_3(r_inner=1, r_outer=5, **args):
        """
        Generate hex-lattice

        Parameters
        ----------
        r_inner : inner radius, integer
        r_outer : outer radius, integer, included

        Return
        ------
        Dict of numpy 2-d arrays of shape (points_per_layer, 2),
        with num_of_layers elements
        coordinated in the last dimension (2 elements) are in
        lattice basis, should multiply trans_visual_3 and trans_simplex_3 to
        change variables to make them visualized or in simplex coordinate.
        """
        unit_sequence = np.array(((0., 1.),
                                  (-1., 0.),
                                  (-1., -1.),
                                  (0., -1.),
                                  (1., 0.),
                                  (1., 1.),
                                  (1., 0.), ),
                                 dtype=np.float64)
        layers = dict()
        cur = unit_sequence[-1] * r_inner

        for r in range(r_inner, r_outer+1):
            ret = []
            for j in range(6):
                for i in range(r):
                    cur += unit_sequence[j]
                    ret += [cur.copy(),]
            layers[r] = np.array(ret)
            cur += unit_sequence[-1]
        return layers

    @staticmethod
    def gen_circ_3(r_inner=1, r_outer=5, **args):
        """
        Generate circle-testpoints

        Parameters
        ----------
        r_inner : inner radius, integer
        r_outer : outer radius, integer, included

        Return
        ------
        Dict of numpy 2-d arrays of shape (points_per_layer, 2),
        with num_of_layers elements
        coordinated in the last dimension (2 elements) are in
        lattice basis, should multiply trans_visual_3 and trans_simplex_3 to
        change variables to make them visualized or in simplex coordinate.
        """
        if "density" in args.keys():
            density = args["density"]
        else:
            density = 6

        if "phase_start" in args.keys():
            p_start = args["phase_start"]
        else:
            p_start = 0

        if "phase_end" in args.keys():
            p_end = args["phase_end"]
        else:
            p_end = 1

        if "endpoint" in args.keys():
            endpoint = args["endpoint"]
        else:
            endpoint = False

        layers = dict()
        for r in range(r_inner, r_outer+1):
            angle = 2 * np.pi / density / r
            ret = []
            for i in np.linspace(r * density * p_start,
                                 r * density * p_end,
                                 density * r, endpoint=endpoint):
                ret += [np.array((r*np.sin(angle*i), r*np.cos(angle*i)), dtype=np.float64)]
            layers[r] = np.matmul(np.array(ret), np.linalg.inv(PriorVariation.trans_visual_3))

        return layers


    config = {3:{"hex" : gen_hex_3.__func__,
                 "circ": gen_circ_3.__func__,},
              4:{}}

    def __init__(self,
                 dim=3,
                 method='hex',
                 matrix=torch.tensor([[0.1, 0.2, 0.7],
                                      [0.3, 0.4, 0.4],
                                      [0.6, 0.3, 0.1]],
                                     dtype=torch.float64),
                 prior=torch.ones(3, dtype=torch.float64)/3,
                 resolution=0.02,
                 r_inner=1,
                 r_outer=2,
                 density=6):
        """
        Initialization
        """
        self.m = 0
        self.n = 0
        self.dim = 0
        self.method = None
        self.set_dimension(dim)
        self.set_method(method)
        self.set_matrix(matrix)
        self.set_prior(prior)
        self.set_perturbations(resolution, r_inner, r_outer, density)
        # self.gen_pts = 

    def set_perturbations(self, resolution=0.02, r_inner=1, r_outer=5, density=6):
        """
        Set resolution / r_inner / r_outer
        """
        self.resolution = resolution
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.density = density

    def set_prior(self, prior):
        """
        Set the prior of teacher
        """
        self.prior = prior

    def get_prior(self):
        """
        Get prior of teacher
        """
        return self.prior

    def set_matrix(self, matrix):
        """
        Set T=L the matrix (joint distribution)
        and the size of n, m
        """
        self.n, self.m = matrix.shape
        self.matrix = matrix

    def get_matrix(self):
        """
        Get matrix of joint distribution
        """
        return self.matrix

    def set_dimension(self, dim):
        """
        Set dimension of prior study.
        """
        if dim not in PriorVariation.config.keys():
            print("Dimension is not set. We can only do dim-3 and 4 cases.")
            return
        self.dim = dim

    def get_dimension(self):
        """
        Get Dimension
        """
        return self.dim

    def set_method(self, method):
        """
        Set method of generating points.
        """
        if self.dim not in PriorVariation.config.keys():
            print("Please set the dim first")
            return

        if method not in PriorVariation.config[self.dim].keys():
            print("Method is not supported, please choose from",
                  PriorVariation.config[self.dim].keys())
            return

        self.method = method

    def get_method(self):
        """
        Get method
        """
        return self.method


    def simulate(self, repeats=100, block_size=100, threads=30, **args):
        """
        Main method

        The `**args` is sent to the method generating sample sets.
        """
        self.tester = st.StabilityTester("cpu")
        self.tester.set_correct_hypo(0)
        self.tester.set_mat_learn(self.matrix)
        self.tester.set_mat_teach(self.matrix)
        self.tester.set_prior_teach(self.prior)
        self.perturb_base = PriorVariation.config[self.dim][self.method](self.r_inner,
                                                                         self.r_outer,
                                                                         density = self.density,
                                                                         **args)
        # posterior = dict()
        self.learn_result = dict()
        self.teach_result = dict()
        for key, value in self.perturb_base.items():
            # posterior[key] = torch.from_numpy(np.matmul(value,
            #                                             PriorVariation.trans_simplex[self.dim]))
            posterior = torch.from_numpy(np.matmul(value,
                                                   PriorVariation.trans_simplex[self.dim]))
            print("Layer:",key, datetime.datetime.today())
            self.learn_result[key] = []
            self.teach_result[key] = []
            # for p_learn in posterior[key]:
            for p_learn in posterior:
                prior_learn = self.prior + p_learn * self.resolution
                self.tester.set_prior_learn(prior_learn)

                inference_result = self.tester.inference_fixed_initial(repeats,
                                                                       block_size,
                                                                       threads,timer=False).numpy()
                self.teach_result[key] += [inference_result[0],]
                self.learn_result[key] += [inference_result[1],]

        with open("Dim"+str(self.dim)+"_"+str(datetime.datetime.today())+".log", "wb") as fp:
            pickle.dump({"base" : self.perturb_base,
                         "teach": self.teach_result,
                         "learn": self.learn_result}, fp)
            pickle.dump({"matrix"    : self.matrix,
                         "prior_t"   : self.prior,
                         "method"    : self.method,
                         "density"   : self.density,
                         "resolution": self.resolution}, fp)

        del self.tester
        return self.teach_result, self.learn_result

    def fastest_path_bin(self, repeats=100, block_size=100, threads=30, delta_angle=0.01):
        """
        In the dim=3 case, use binary search on each level to find out
        worst / best position on each circle.
        """
        assert self.dim == 3
        self.tester = st.StabilityTester("cpu")
        self.tester.set_correct_hypo(0)
        self.tester.set_mat_learn(self.matrix)
        self.tester.set_mat_teach(self.matrix)
        self.tester.set_prior_teach(self.prior)

        self.learn_result = dict()
        self.teach_result = dict()

        fastest_path = []

        ave = lambda x, y: (x + y) / 2
        for radius in range(self.r_inner, self.r_outer+1):
            threshold = delta_angle / radius
            # left and right nodes
            data = dict()
            left = 0.
            right = 1.
            mid = ave(left, right)
            coords = PriorVariation.gen_circ_3(1, 1, density=3,
                                               phase_start=left,
                                               phase_end=right,
                                               endpoint=True)
            prior_learn = torch.from_numpy(np.matmul(coords[1],
                                                     PriorVariation.trans_simplex_3))
            prior_learn = prior_learn * radius * self.resolution + self.prior

            tmp = []
            for prior in prior_learn:
                self.tester.set_prior_learn(prior)
                inference_result = self.tester.inference_fixed_initial(repeats,
                                                                       block_size,
                                                                       threads)
                tmp += [inference_result[1][0],]

            data[left]  = tmp[0]
            data[mid]   = tmp[1]
            data[right] = tmp[2]

            # Very typical bisection method. Looks like a bad implementation.
            while right - left > threshold:
                q1 = ave(left, mid)
                q3 = ave(mid, right)
                coords = PriorVariation.gen_circ_3(1, 1, density=2,
                                                   phase_start=q1,
                                                   phase_end=q3,
                                                   endpoint=True)
                prior_learn = torch.from_numpy(np.matmul(coords[1],
                                                         PriorVariation.trans_simplex_3))
                prior_learn = prior_learn * radius *self.resolution + self.prior
                self.tester.set_prior_learn(prior_learn[0])
                data[q1] = self.tester.inference_fixed_initial(repeats,
                                                               block_size,
                                                               threads)[1][0]
                self.tester.set_prior_learn(prior_learn[1])
                data[q3] = self.tester.inference_fixed_initial(repeats,
                                                               block_size,
                                                               threads)[1][0]
                for i in range(2):
                    if data[left] > data[right]:
                        left = q1
                        q1 = mid
                    else:
                        right = q3
                        q3 = mid
                mid = ave(left, right)

            # Now data[mid] can reflects the steepest point we have on the circle.
            print(data)
            fastest_path += [(mid, data[mid]),]
            del data
        return fastest_path


    @staticmethod
    def read_data(filename):
        """
        Basic method of reading files.
        """
        setup = None
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            try:
                setup = pickle.load(fp)
                print("Loaded a new version log file.")
            except EOFError:
                print("Loaded an old version log file.")
        return data, setup


if __name__ == '__main__':
    pass
