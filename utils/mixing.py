import random
import torch
import numpy as np

def classmix(A, B, Sa, Sb):
    C = Sa.unique()
    rand_indices = torch.randperm(C.size(0))[:C.size(0)//2]
    c = C[rand_indices]

    mask = torch.isin(Sa, c)

    Xa = mask * A   + ~mask * B
    Ya = mask * Sa  + ~mask * Sb

    return Xa, Ya

def cutmix(X, Y):
    batch_size, _, H,W = X.shape

    idx = torch.randperm(X.size(0))
    Xs, Ys = X[idx, :,:,:], Y[idx, :,:]

    # lam = np.random.uniform()
    lam = np.random.beta(0,1)
    rx = np.random.uniform(0,W)
    ry = np.random.uniform(0,H)
    rw = np.sqrt(1-lam)
    rh = np.sqrt(1-lam)
    x1 = np.rint(np.clip((rx-rw) / 2, min=0)).astype(np.int32).item()
    x2 = np.rint(np.clip((rx+rw) / 2, max=W)).astype(np.int32).item()
    y1 = np.rint(np.clip((ry-rh) / 2, min=0)).astype(np.int32).item()
    y2 = np.rint(np.clip((ry+rh) / 2, max=H)).astype(np.int32).item()
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)

    # import pudb
    # pu.db

    X[:,:, x1:x2, y1:y2] = Xs[:,:, x1:x2, y1:y2]
    Y = lam * Y + (1 - lam) * Ys

    return X, Y
