import numpy as np
import pulp as lp

#-------------------------------------------------------------------------------------------------------------
def Generate_f(Nj,w,indc):
  #creates 1xNj dimensional utility generanting mechanism
  #indc =1 -- Marginal contribution(w), indc =2 -- Equal share(w), indc =3 for all 1
  f= np.zeros((Nj,1))
  for j in range (Nj):
      if indc == 1:
          val =0
          if j>=1:
              val = w[j-1]
          f[j] = w[j]-val;
      elif indc ==2:
              f[j] = w[j]/(j+1)
      else:
          f=np.ones((Nj,1))
  return np.transpose(f)

#-------------------------------------------------------------------------------------------------------------
def LinearProgram(A,b,c):
  #solves Linear program min c^T x subject to Ax<=b

  m = A.shape[0]     # no. of constraints
  n = A.shape[1]     # no. of variables

  if c.shape[0] != n or c.shape[1]!=1:
    raise Exception('c must be a column vector of length',n,'Current dimenstions are',c.shape,'should be', (n,1))

  if b.shape[0] != m or b.shape[1]!=1:
    raise Exception('b must be a column vector of length',m,'Current dimenstions are',b.shape,'should be', (m,1))

  set_m = range(0,m )
  set_n = range(0,n)
  x = lp.LpVariable.dicts("x", set_n, cat='Continuous')
  prob = lp.LpProblem("numpy_constraints", lp.LpMinimize)
  prob += lp.lpSum([c[i]*x[i] for i in set_n])

  for i in set_m:
    prob += lp.lpSum([A[i][j]*x[j] for j in set_n]) >= b[i]
  prob.solve()
  x_soln = np.array([x[i].varValue for i in set_n]) 
  z = prob.objective.value()
  return z,x_soln

#-------------------------------------------------------------------------------------------------------------
def GenerateIR(kappa):
    idx =1
    kk = kappa+1
    for a in range(kk):
        for x in range(kk-a):
            for b in range(kk-a-x):
                if a+x+b == kappa or a*x*b ==0:
                    n_row = np.array([a,x,b])
                    if idx ==1:
                        matrix = n_row
                        idx = idx+1
                    else:
                        matrix = np.vstack((matrix,n_row))
    return matrix

def concatenate(matrix1,matrix2):
    r1,c1 = matrix1.shape[0], matrix1.shape[1] 
    r2,c2 = matrix2.shape[0], matrix2.shape[1] 
    new_mat = np.zeros((r1*r2,c1+c2))
    rval =0
    for l1 in range(r1):
        for l2 in range(r2):
            new_mat[rval + l2,0:c1]       = matrix1[l1,:]
            new_mat[rval + l2,c1:c1+c2]   = matrix2[l2,:]
        rval = rval+r2
    return new_mat

def ArbtitraryIR(C):
    k = C.shape[0]  
    for j in range(k):
        mat0 = GenerateIR(C[0])
        mat1 = GenerateIR(C[1])
        tempIR = concatenate(mat0,mat1)
        if k==3:
            mat2 = GenerateIR(C[2])
            tempIR = concatenate(tempIR,mat2)
    IR = tempIR[1:,:]
    return IR

#-------------------------------------------------------------------------------------------------------------





  




  