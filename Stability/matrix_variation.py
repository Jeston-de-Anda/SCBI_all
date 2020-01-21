#!/usr/bin/env python
"""
Matrix Variation
"""

import pickle
from datetime import datetime as dt
import numpy as np
import torch
import sinkhorn_torch as sk
import stability_testers as st
import equi_roc_mat_perturb as er






class MatrixVariation:
    """
    Matrix Variation
    """
    PREFIX = ""

    @staticmethod
    def generate_method_mix_roc(matrix, index=2, density=20, lower=0, upper=1):
        """
        Generate Mix RoC Matrix

        `matrix[:, index] = (x) * matrix[:, c_hypo] + (1-x) * matrix[:, index]`
        The above new column is guaranteed to be normalized, if `matrix` is col-normalized
        """
        epsilon = 1e-7
        c_hypo = 0 # correct hypothesis, by default 0
        mat = matrix.clone()
        ref = mat[:, index]/(mat[:, index] - mat[:, c_hypo])
        ref[ref > 0] = -np.inf
        lower_limit = torch.max(ref).item()
        # lower_limit = torch.max(mat[:, index]/(mat[:, index] - mat[:, c_hypo])).item()
        lower = max(lower_limit + epsilon, lower)

        parameters = torch.linspace(lower, upper, density, dtype=torch.float64)
        result = []
        for ratio in parameters:
            mat = matrix.clone()
            mat[:, index] = ratio * mat[:, c_hypo] + (1 - ratio) * mat[:, index]
            result += [mat,]
        return result

    @staticmethod
    def generate_method_equi_roc(matrix, density=90):
        """
        This will change matrix itself
        """
        index = er.adjust_matrix(matrix)

        return er.generate_single(matrix, index, density)

    @staticmethod
    def generate_method_single_entry(matrix, coordinate=(0, 0), magnitude=0.1, size=10):
        """
        Single Entry Perturbation

        coordinates of the form (i, j) after a perturbation
        we do a column-normalization
        """
        magnitude = min(magnitude, matrix[coordinate])
        parameters = np.linspace(-magnitude, magnitude, size * 2 + 1)

        results = []
        tmp = torch.zeros_like(matrix)
        for par in parameters:
            if par == 0:
                continue
            tmp[coordinate] = par
            results += [sk.col_normalize(matrix + tmp,
                                         torch.ones(matrix.shape[1], dtype=torch.float64)),]

        return results

    @classmethod
    def set_prefix(cls, prefix):
        """
        Set MatrixVariation.PREFIX
        """
        cls.PREFIX = prefix

    def __init__(self, device="cpu",
                 generate_method=generate_method_single_entry,
                 brand_name=""):
        """
        Parameters
        ----------
        generate_method: method to generate perturbations,
                         see `MatrixVariation.set_generate_method`
        """
        self.set_device(device)
        self.tester = st.StabilityTester(device)
        self.correct_hypo = 0
        self.set_generate_method(generate_method)
        self.set_brand_name(brand_name)



    def set_brand_name(self, name):
        """
        Set Brand Name
        """
        self.brand_name = name

    def set_device(self, device):
        """
        Set Device
        """
        self.device = device

    def get_device(self):
        """
        Get Device
        """
        return self.device

    def set_generate_method(self, method):
        """
        Set Generate Method

        Parameters
        ----------
        method: a function or method of the format:
                prior, [matrix1, ...] method(matrix, prior, *args)
        """
        self.generate_method = method


    def run_single(self, mat_data, prior, args=(200, 50, 40)):
        """
        Run with single teaching matrix and single prior
        """
        self.tester.set_prior_learn(prior)
        self.tester.set_prior_teach(prior)
        mat_teach = mat_data[0]
        self.tester.set_mat_teach(mat_teach)
        mats_learn = mat_data[1]

        results = []
        for mat in mats_learn:
            self.tester.set_mat_learn(mat)
            results += self.tester.inference_fixed_initial(*args)

        with open(MatrixVariation.PREFIX + self.brand_name + str(dt.today()) + ".log",
                  "wb") as f_ptr:
            pickle.dump({"mat_teach" : mat_teach,
                         "prior"     : prior,
                         "mats_learn": mats_learn,
                         "results"   : results}, f_ptr)

    def run(self, setup_filename, generate=False, **args):
        """
        Main entry
        """
        if not generate:
            # setup file contains {"mats": [(mat_teach, [mat_learn1, mat_learn2, ...]), ...],
            #                      "priors": priors}
            with open(setup_filename, "rb") as f_ptr:
                data = pickle.load(f_ptr)
                mats = data["mats"]
                priors = data["priors"]
        else:
            # setup file contains only {"mats": list of mat_teach, "priors": priors}
            with open(setup_filename, "rb") as f_ptr:
                data = pickle.load(f_ptr)
            mat_d = data["mats"]
            priors = data["priors"]
            mats = []
            for mat in mat_d:
                mats += [(mat, self.generate_method(mat, **args)), ]

        self.tester.set_correct_hypo(self.correct_hypo)

        for mat_data in mats:
            for prior in priors:
                self.run_single(mat_data, prior)



if __name__ == '__main__':
    MODEL = MatrixVariation("cpu")
    MODEL.run("Dim3_setup.log", generate=False)
