'''
In CNNs images are stored as:
(N, C, H, W)
N: number of images
C: channels, 1 = grayscale 3 = RGB
H: height
W: width
'''

'''
A kernel is a small matrix of no.s used to extract specific patterns
from an img
eg, one designed to detect texture/ edges, as it slides over images
it produces a higher value when it reaches the required pattern vs a 
lower value when it does not
These kernel values are not defined, they are learned during training 
and each kernel captures useful features for the task

A kernel can only be placed in exaclty one position because it must fully fit in an img
when applied, it computes a single number
the entire region is compressed to one scalar value

in convolution, ,
output size = (input size - kernel size) + 1 (stride)

Losses that occur:
Spatial resolution loss
    output becomes smaller
    fewer positions representing the image
    Fine grained details disappear 
Compression within each patch
    each patch is reduced to a single number
    different patches can give same output even if internal pixel arrangements are diff
    ie some distinctions are lost
    
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

def conv2d_backward(X, kernel, dout):
    # X: input img
    # kernel: (k, k)
    # dout: gradient from next layer
    
    k = kernel.shape[0]
    
    # Step 1: Recreate patches from forward pass
    # shape: (num_patches, k*k)
    patches = im2col(X, k)
    
    # Step 2: Faltten dout to match patches
    # (H_out, W_out) -> (num_patches,)
    dout_flat = dout.flatten()
    
    # Gradient wrt kernel
    # each patch contributes to kernel update
    # shape: (k*k,)
    dkernel_flat = patches.T @ dout_flat
    # sccumulates how each patch affects the kernel
    
    # reshape back to (k, k)
    dkernel = dkernel_flat.reshape(k, k)
    
    # Gradient wrt input
    # reversing the forward operations
    
    # flatten kernel
    kernel_flat = kernel.flatten()
    
    # Computing gradient for each patch
    # shape: (num_patches, k*k)
    dpatches = np.outer(dout_flat, kernel_flat)
    
    # Step4: reconstruct dx from patches
    H, W = X.shape
    dX = np.zeros_like(X)
    
    patch_idx = 0
    
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            
            # reshape back to (k, k)
            patch_grad = dpatches[patch_idx].reshape(k, k)
            
            # add gradient to correct location
            dX[i:i+k, j:j+k] += patch_grad
            
            patch_idx += 1
    
    return dX, dkernel
'''
backwad pass:
how did each part of the input and kernel contribute to the final loss
during the forward pass:
conv operation takes patches + kernel -> output: feature map

loss is computed at the end of the network 
the backward process begins by propagating gradients from loss toward the input

the conv layer receives is dout, represents how much each output value effects the loss
for every position in the output feature map, dout specifies how sensitive loss is to the change of
specific output value

To compute gradients, the backward pass must follow the same structure as forward pass
as it conceptually maps patches of the input to the output, the backward pass
reverses this relationship, each output traces the patch that produced it

gradient wrt kernel is computed by looking at how each patch influences the output
if an output has a large gradient -> strongly affects loss
then the patch that produces it should have a stronger influence on the kernel
the kernel gradient is an accumulation of al patches, where each patch is weighed by how imp the putput was
thus all patches are combined with their resp dout values
dout tells how each kernel weight should reduce the loss

gradient wrt input
each output value depends on all elements inside the pach
during backward pass he gradient of that output much be dist back to all those inputs
this is guided by kernel weights
in forward pass, each input value was multiplied by a specific kernel weight
thus in backward pass the influence flows back proportionately through the same weights

patches overlap in the input ie
a single inpt pixel may have contributed to mutl outputs during the forward pass
thus in the backward pass the pixel will receive gradient contri from mutl outpts
these contris are summed together which is why the input gradientt is built by accumulating values rather than 
assigning at once

overall
the backwad process reverses the compression done in formward pass
the forward pass reduces eacch patch to a single number
the backward pass expands each output gradient back to its corresponding patch
and conbines all overlapping contri at the same time, aggregarting info rom all patches
to determine how kernels sshould be adjustd
this ensures both input + kernel receive precise updates based on their inflience on the final loss

'''


# ReLU Activation
# ReLU(x) = max(0, x)
def relu_forward(X):
    # Replace negative values with 0
    out = np.maximum(0, X)
    return out

# Backward ReLU:
# input > 0 -> gradient passes
# input < 0 -> gradient = 0

def relu_backward(X, dout):
    # d out: gradient from the next layer
    dX = dout.copy()
    
    # Zero gradient where input was negative
    
    dX[X <= 0] = 0
    
    return dX

# flattening

# forward: 2D -> 1D
def flattening_forward(X):
    # save org shape
    original_shape = X.shape
    
    # Convert multi-dim to 1D vector
    out = X.flatten()
    
    return out, original_shape

# backward: 1D gradient -> reshape -> 2D
# it doesnt change values
# it passes gradient in correct shape
def flattening_backward(dout, original_shape):
    # Restoring gradient back to original shape
    dX = dout.reshape(original_shape)
    
    return dX

