import numpy as np

denseRep = np.load("denseArray.npy")
norm_dense_rep = denseRep-np.mean(denseRep, axis = 0)

cov_mat=np.cov(norm_dense_rep, rowvar=False)

values, vectors = np.linalg.eigh(cov_mat)
indices = np.argsort(values)[::-1]
values = values[indices]
vectors = vectors[:,indices]
np.save("eigenvalues.npy",values)
np.save("eigenvectors.npy",vectors)
