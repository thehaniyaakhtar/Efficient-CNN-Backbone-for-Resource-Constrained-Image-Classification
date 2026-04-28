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
    
    