#include "test.h"


int solve_quadratic_equation(double A, double B, double C, double* result)
{
  /*
    Solve equation $A x^2 + B x + C = 0$, use result[4] to pass result, 
    format is 
      $x_1 = result[0] + result[1] I$,
      $x_2 = result[2] + result[3] I$.
   */
  if(A==0 && B==0){
    if(C) return -1;
    for(int i=0; i<4; i++){
      *(result+i)=0;
    }
    return 1;
  }
  if(A==0){
    *result = *(result+2) = -C/B;
    *(result+1) = *(result+3) = 0;
    return 1;
  }
  double discriminant = B*B-4.0*A*C;
  if(discriminant<0){
    double im = sqrt(-discriminant)/2.0/A;
    double re = -B/2/A;
    *result = *(result+2) = re;
    *(result+1) = im;
    *(result+3) = -im;
    return 2;
  }
  if(discriminant==0){
    *(result+1) = *(result+3) = 0;
    *result = *(result+2) = -B/2/A;
    return 3;
  }
  double var = sqrt(discriminant)/2.0/A;
  *(result+1) = *(result+3) = 0;
  *result = -B/2/A-var;
  *(result+2) = -B/2/A + var;
  return 4;
}

int sinkhorn2(double matrix[2][2], double col_sum[2],
	      double row_sum[2], double result[2][2]) {
  /*
    Given matrix=
      [[a, b],
       [c, d]],
    find the scaled matrix whose column sum is col_sum
    and row sum is row_sum, in place of result.
   */
  double A, B, C;
  A = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
  C = row_sum[0]*col_sum[0]*matrix[0][0]*matrix[1][1];
  B = -col_sum[0] * A - row_sum[1]*matrix[0][1]*matrix[1][0] - row_sum[0]*matrix[0][0]*matrix[1][1];

  double solution[4];
  double r; // upper-left entry of result.
  if(solve_quadratic_equation(A,B,C,solution) < 3) return -1;
  if(solution[0]>=0 && solution[0]<=col_sum[0] && solution[0]<=row_sum[0]){
    r = solution[0];
  } else {
    r = solution[3];
  }

  result[0][0] = r;
  result[0][1] = row_sum[0] - r;
  result[1][0] = col_sum[0] - r;
  result[1][1] = row_sum[1] - result[1][0];
  
  // test
  // printf("%lf+(%lf)I\n%lf+(%lf)I\n",solution[0],solution[1],solution[2],solution[3]);
  // test
  
  return 0;
}

int sinkhornc(double matrix[2][2], double col_sum[2],
	      double result[2][2]) {
  double row_sum[2]={1.0,1.0};
  return sinkhorn2(matrix,col_sum,row_sum,result);
}



int teaching(double T[2][2], double L[2][2],
	     double pT, double pL, double T_res[2], double L_res[2])
{
  /*
    T and L be two original joint distributions, they are defined as global 
    variables, could be used directely.
   */
    double result[2][2];
    double col_sum[2];
    col_sum[0] = pT * 2.0;
    col_sum[1] = 2.0 - pT * 2.0;
    sinkhornc(T, col_sum, result);
    T_res[0] = result[0][0];
    T_res[1] = result[1][0];

    col_sum[0] = pL * 2.0;
    col_sum[1] = 2.0 - pL * 2.0;
    sinkhornc(L, col_sum, result);
    L_res[0] = result[0][0];
    L_res[1] = result[1][0];
    return 0;
}
