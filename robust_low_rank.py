import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as lasp
from sklearn.decomposition import SparsePCA
from mw import solveGroth

def compute_robust_low_rank(data, total_k, range_kprime, d):
    # PCA + sparse PCA
    Count = 0
    Count += 1
    projection_error = []
    sdp_val = []
    minPi = []
    min_sdp_val = 1000000

    for k in range_kprime:
        print("processing k=", k)
        eigs, eigvecs = lasp.eigsh(data, k=k, which='LA', tol=0.00001)
        Pi = np.matmul(eigvecs, eigvecs.T)
        projected_data = np.matmul(Pi, np.matmul(data, Pi))

        if k < total_k:
          spca = SparsePCA(n_components=total_k-k, random_state=0, alpha=1e-5, normalize_components=True)

          spca.fit(100*(data-projected_data))

          u = spca.components_
          A = np.matmul(np.eye(d)-Pi,np.matmul(u.T,u))
          B = np.matmul(A, np.eye(d)-Pi)
          eigval, U = lasp.eigsh(B, k=total_k-k, which='LA', tol=0.00001)

          D = 1.0*np.diag(eigval > 0.00001)
          U = np.matmul(U,D)

          sPi = Pi + np.matmul(U, U.T)
        else:
          sPi = Pi

        projected_data = np.matmul(sPi, np.matmul(data, sPi))
        projection_error.append(np.trace(data) - np.trace(projected_data))
        [curr_y, min_val, curr_alpha, avg_y_val] = solveGroth(sPi,d)
        sdp_val.append(min_val)
        minPi.append(sPi)
    return [projection_error, sdp_val, minPi, min_sdp_val]
