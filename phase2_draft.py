# pooling
# reduces size of feature maps
# keeps imp info

import numpy as np

'''
input:
1 3 2 4
5 6 1 2
7 2 8 3
4 5 9 1

2x2 maxpool:
[1 3 | 2 4] → max = 3, 4
[5 6 | 1 2] → max = 6, 2

[7 2 | 8 3] → max = 7, 8
[4 5 | 9 1] → max = 5, 9

output:
3 4
6 2
7 8
5 9

for each window only the max survives
'''

def maxpool_forward(X, k, stride):
    # k: window size
    # stride: step size
    
    H, W = X.shape
    
    out_H = (H - k) // stride + 1
    out_W = (W - k) // stride + 1
    
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            
            # define window
            h_start = i * stride
            h_end = h_start + k
            
            W_start = j * stride
            W_end = W_start + k
            
            # extract patch
            patch = X[h_start : h_end, W_start : W_end]
            
            # take max
            out[i, j] = np.max(patch)
            
    return out

# MaxPool Backward:
# Find max location
# Put gradient there
# Everything else = 0

def maxpool_backward(X, dout, k, stride):
    # X: org input (H, W)
    # dout: gradieny from next layer (H_out, W_out)
    
    H, W = X.shape
    dX = np.zeros_like(X)
    
    out_H, out_W = dout.shape
    
    for i in range(out_H):
        for j in range(out_W):
            
            # define window
            h_start = i * stride
            h_end = h_start + k
            
            w_start = i * stride
            w_end = w_start + k
            
            # extract patch
            patch = X[h_start:h_end, w_start:w_end]
            
            max_idx = np.argmax(patch)
            
            max_pos = np.unravel_index(max_idx, patch.shape)
            
            dX[h_start + max_pos[0], w_start + max_pos[1]] += dout[i, j]
    
    return dX       

# average pool

def avgpool_forward(X, k, stride):
    H, W = X.shape
    
    out_H = (H - k) // stride
    out_W = (W - k) // stride
    
    out = np.zeros((out_H, out, W))
    
    for i in range(out_H):
        for j in range(out_W):
            
            h_start = i * stride
            h_end = h_start + k
            
            w_start = j * stride
            w_end = w_start + k
            
            patch = X[h_start:h_end, w_start:w_end]
            
            # take avg
            out[i, j] = np.mean(patch)
            
    return out


def avgpool_backward(X, dout, k, stride):
    H, W = X.shape
    dX = np.zeros_like(X)
    
    out_H, out_W = dout.shape
    
    for i in range(out_H):
        for j in range(out_W):
            
            h_start = i * stride
            h_end = h_start + k
            
            w_start = j * stride
            w_end = w_start + k
            
            gradient = dout[i, j] / (k * k)
            
            dX[h_start:h_end, w_start:w_end] += gradient
            
    return dX