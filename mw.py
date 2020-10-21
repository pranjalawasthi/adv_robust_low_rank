import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as lasp
import time
import math

def updateWeights(w,x,threshold,eps, N):
    for i in range(len(x)):
        if np.abs(x[i])>threshold:
            w[i]*=(1+eps)
    s = np.sum(w)

    w = np.divide(w,s)
    w *= N
    return w

def updateWeights_2(w,x,threshold,eps, N):
    cap = 4
    for i in range(len(x)):
        if np.abs(x[i])>threshold:
            w[i]*= math.exp(eps*min(x[i]**2-1, cap))
    s = np.sum(w)

    w = np.divide(w,s)
    w *= N
    return w

def checkCondition(x,threshold):
    return not (np.max(np.abs(x))>threshold)


def solveGroth(A, n, init_val=None):
    """
    ...
    Parameters
    ----------
    A: np.matrix
        dfajdslkf
    n: int
        ddddddd
    init_val: float, optional
        dsfdsafdasfd
    Returns
    -------
    list of
        float:
        float:
        float:
        float:
    """
    eps=0.5
    eta=0.05
    threshold = 1.1
    N = n
    Ap = A
    if init_val is not None:
      w=init_val
    else:
      w = np.ones(N)
    min_val = np.sum(np.abs(Ap))
    curr_y = np.zeros(N)
    curr_alpha = np.zeros(N)
    avg_y_val = np.zeros(N)
    avg_X=np.zeros((N,N))
    avg_alpha=np.zeros(N)

    T=4*N
    schedule_size = round(T/8) #We change eps in epochs

    print("iteration bound:",T)
    z = np.zeros(N) 
    vals = 0
    g = np.random.standard_normal(T)
    for i in range(T):
        if (i+1)%(T/2)==0:
            eps=0.01
        if i%schedule_size ==0 and eps>0.01 :
            eps=eps/2
        if i%schedule_size == 0 and i>T/2:
             eps /= 2



        wtil=(1-eta)*w+eta*np.ones(N)
        w1 = np.array([1/np.sqrt(j) for j in wtil])
        start_time = time.time()
        d = np.tile(w1, (N, 1))
        M = np.multiply(Ap,d)
        d = np.tile(np.array([w1]).transpose(), (1,N))
        M = np.multiply(M,d)

        start_time = time.time()
        eigval, eigvec = lasp.eigsh(M, k=1, which='LA', tol=0.00001)


        y = eigvec[:,0] 
        y *= np.sqrt(N)
        y = np.multiply(y,w1) 
        avg_y_val += y**2
        val = np.matmul(np.transpose(y), np.matmul(Ap,y))
        avg_alpha+=val*w


        if val < min_val:
          min_val = val
          curr_y = y
          curr_alpha = w

        vals += val 
        print("iterate", i, "val = ", val, " minval=", min_val, " linf of curr y=", np.max(np.abs(y**2)) , " infinity norm avg X =", np.max((1.0/(i+1))*avg_y_val), "SDP sol val:", vals/(i+1), "eps,eta=", eps, " , ", eta)


        if checkCondition(y,threshold):
            print(y,"Current iterate Condition satisfied, i : ",i)
            print("min val = ", min_val)
            print("curr_y = ", curr_y)
            print("curr_alpha = ", curr_alpha)
            print("inf norm of curr_y=", max(abs(curr_y)))
            return [np.matmul(curr_y,curr_y.T),min_val, curr_alpha, avg_y_val] 
        elif checkCondition((1.0/(i+1))*avg_y_val, threshold):
            avg_y_val=(1.0/(i+1))*avg_y_val
            avg_val = vals/(i+1)
            print(avg_y_val," Avg Condition satisfied, i : ",i)
            print("min val = ", min_val)
            print("curr val=", avg_val)
            print("curr_alpha = ", (1.0/i)*avg_alpha)
            print("inf norm of avg_y=", max(abs(avg_y_val)))
            return [(1.0/(i+1))*avg_X,min_val, curr_alpha, avg_y_val] 


        if i < T/2:
          w = updateWeights_2(w,y,threshold, eps, N)
        else:
          w = updateWeights(w,y,threshold, 2*eps, N)
        u = y*g[i]
        z += u
    print("min val = ", min_val)
    print("sum of curr_alpha = ", sum(curr_alpha))
    print("sum weights at end = ", sum(w))
    print("inf norm of curr_y=", max(abs(curr_y)))

    return [np.matmul(curr_y, curr_y.T), min_val, curr_alpha, avg_y_val] 
