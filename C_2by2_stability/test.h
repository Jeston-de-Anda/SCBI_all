#ifndef __STABILITY_TEST_H
#define __STABILITY_TEST_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern int teaching(double T[2][2], double L[2][2],
                    double pT, double pL,
                    double T_res[2], double L_res[2]);

extern int sinkhornc(double matrix[2][2], double col_sum[2],
                     double result[2][2]);

extern int sinkhorn2(double matrix[2][2], double col_sum[2],
                     double row_sum[2], double result[2][2]);

extern int solve_quadratic_equation(double A, double B,
                                    double C, double* result);
#endif
