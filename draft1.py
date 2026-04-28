'''
In CNNs images are stored as:
(N, C, H, W)
N: number of images
C: channels, 1 = grayscale 3 = RGB
H: height
W: width
'''

import numpy as np
# example of a grayscale image:
X = np.array([
    [1, 2 ,3],
    [4, 5, 6],
    [7, 8, 9]
])

# CNNs are stored as (N, C, H, W)
# shape: (1, 1, 3, 3)
# padding it by applying function
# padding enables information losing
# 3x3 image + 3x3 filter -> output = 1x1
# with padding 5x5 + 3x3 filter -> output = 3x3

'''
image dimensions: 3x3
filter: 2x2
no padding
stride = 1

how many positions can the filter slide over:
width = 3
filter = 2
positions = 3 - 2 + 1 = 2

height = 3
filter = 2
postions = 3 - 2 + 1 = 2

positions = 2 x 2 = 4
'''

# building padding funcions:
def pad2d(X, pad):
    # X is input image
    X_padded = np.pad(
        # np.pad adds values around the edges of array
        X_padded = np.pad(
            X,
            (
                (0, 0),     # do not pad batch dimension
                (0, 0),     # do not pad pad channels          
                (pad, pad), # Add padding to top and bottom
                (pad, pad)  # add padding to left and right
            ),
            mode = 'constant' # fill added values wth 0
        )
    )
    return X_padded
    # return padded image
    
# for input (1, 3, 5, 5), pad = 1
# padddinf adds +2 to height (top + bottom)
# padding adds +2 to width (left + right)
'''
1 2 3
4 5 6
7 8 9

padding:
0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0
'''

# Im2Col

'''
image:
1 2 3
4 5 6
7 8 9

small box:
1 0
0 1

Goal: produce a smaller image

Processed in parts:

Image   Kernel:
1 2     1 0
4 5     0 1

Multiplied:
(1×1) + (2×0) + (4×0) + (5×1)
= 1 + 0 + 0 + 5
= 6
One number in output

Moved the box to right, then down
this process is not possible to code:
nested loops are slow
hard to scale
pain for backprop

this im2col
Step 1: Each row = one patch
[1 2 4 5]
[2 3 5 6]
[4 5 7 8]
[5 6 8 9]

Step 2: 
Flatten Kernel
[1 0 0 1]

Step 3: Multiply  in one shot
matrix(4x4) x vector(4x1) = (4x1)
'''

def im2col(X, k):
    # X is a 2d image (H, W)
    # k is kernel size
    H, W = X.shape
    # extract H and W of image
    
    patches = []
    # expty list to store all patches
    
    # slide window vertically
    for i in range(H - k + 1):
        # slide window horizontally
        # H-k+1 as pach must fit inside image
        for j in range(W - k + 1):
            # extract k x k patch
            # controls left to right sliding
            patch = X[i:i+k, j:j+k]
            # slicing from i/j to i+k/j+k
            
            # flatten to 1D
            patch_flat = patch.flatten()
            
            
            #store
            patches.append(patch_flat)
            
    return np.array(patches)
    # convert patch to array
    
# image = 5 x 5
# kernel = 3 x 3 
# 5 - 3 + 1 = 3; 3 vert and hor posiitons
# 3 x 3 = 9 rows in im2col output

# single channel conv2d
# output = patches @ kernel_flat
# where patch x kernel -> oen number
def conv2d_forward(X, kernel):
    k = kernel.shape[0]
    
    # convert image to patches
    # using previous kernels
    patches = im2col(X, k)
    
    # flatten kernel
    # it can be multiplied with each patch
    kernel_flat = kernel.flatten()
    
    # matrix multiply
    # actual convolution
    # 
    out = patches @ kernel_flat
    
    # reshape to 2d
    # convert 1D output to 2D
    out_size = int(np.sqrt(len(out)))
    out = out.reshape(out_size, out_size)
    
    return out

# backward 
