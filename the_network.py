import numpy as np

# Padding
def pad2d(X, pad):
    # Add zero padding around H and W dimensions
    X_padded = np.pad(
        X, # input tensor (N, C, H, W)
        (
            (0, 0),     # No padding for batch dimension
            (0, 0),     # No padding for channel dimension
            (pad, pad), # Pad top and bottom
            (pad, pad)  # Pad left and right
        ),
        mode = 'constant' # Fill padded values with 0
    )
    return X_padded # Return padded tensor 

# Im2col
def im2col(X, k):
    H, W = X.shape # Get H and W of input image
    patches = []   # Store flattened patches
    
    # Slide kernel vertically
    for i in range(H - k + 1):
        # Slide kernel Horizontally 
        for j in range(W - k + 1):
            patch = X[i:i+k, j:j+k]
            patch_flat = patch.flatten()
            patches.append(patch_flat)
            
    return np.array(patches)
    # convert list to numpy array
    
# Convolution Forward

def conv2d_forward(X, kernel):
    k = kernel.shape[0]
    
    patches = im2col(X, k)
    # an area that a kernel looks over at a time
    kernel_flat = kernel.flatten()
    # 3x3 kernel becomes 9x1 vector
    
    out = patches @ kernel_flat
    # each row of patches represents one patch
    # dot product with kernel
    # gives one number per patch
    
    out_size = int(np.sqrt(len(out)))
    out = out.reshape(out_size, out_size)
    # out is a flat vector of length 9
    # reshape 9 to 3x3
    
    return out

# Convolution Backward

def conv2d_backward(X, kernel, dout):
    # dout is gradient of loss wrt output
    k = kernel.shape[0]
    
    patches = im2col(X, k)
    dout_flat = dout.flatten()
    # each patch produced one output
    # each dout_flat value tells how imp was the patch's output for the loss
    
    dkernel_flat = patches.T @ dout_flat
    # gradient wrt kernel
    # for each patch, if it 
    dkernel = dkernel_flat.reshape(k, k)
    
    # gradient wrt input
    kernel_flat = kernel.flatten()
    dpatches = np.outer(dout_flat, kernel_flat)
    
    H, W = X.shape
    dX = np.zeros_like(X)
    
    patch_idx = 0
    
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            patch_grad = dpatches[patch_idx].reshape(k, k)
            dX[i:i+k, j:j+k] += patch_grad
            patch_idx += 1
            
    return dX, dkernel
    
    