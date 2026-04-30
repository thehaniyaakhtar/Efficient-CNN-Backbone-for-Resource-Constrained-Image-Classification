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

# ReLU

def relu_forward(X):
    return np.maximum(0, X)

def relu_backward(X, dout):
    dX = dout.copy() # Copy the upstream gradient
    dX[X <= 0] = 0 # Zero out where input is negative
    return dX
    
# Flatten

def flatten_forward(X):
    original_shape = X.shape # store org shape
    out = X.flatten() # convert to 1D vector
    return out, original_shape # Return flattened + shape

def flatten_backward(dout, original_shape):
    return dout.reshape(original_shape) # Restore original shape


# Maxpool

def maxpool_forward(X, k, stride):
    H, W = X.shape # Input dimensions
    
    out_H = (H-k) // stride + 1 #  Output height
    out_W = (W-k) // stride + 1 # Output width
    # Computing how many positions the window can take
    # the windoe much fully fit inside the input
    
    out = np.zeros((out_H, out_W)) # Initialize output
    # an empty output grid
    
    for i in range(out_H):
        for j in range(out_W):
            # sliding the window across the image
            h_start = i * stride
            h_end = h_start + k
            
            w_start = j * stride
            w_end = w_start + k
            
            patch = X[h_start:h_end, w_start:w_end] 
            # Extract patch that the window is looking for
            out[i, j] = np.max(patch) # Take max from the selcted patch
        
    return out # Return pooled output


def maxpool_backward(X, dout, k, stride):
    H, W = X.shape
    dX = np.zeros_like(X)
    # initializing the gradient for the input
    
    out_H, out_W = dout.shape
    # conputing the number of output positions
    
    for i  in range(out_H):
        for j in range(out_W):
            # looping over each output position, each (i, j) corresponds to one patch in forward
            h_start = i * stride
            h_end = h_start + k
            
            w_start = j * stride
            w_end = w_start + k
            
            patch = X[h_start:h_end, w_start:w_end]
            # extract the same region used in forward
            
            max_idx = np.argmax(patch) # index of max value
            # argmax finds the largest value
            max_pos = np.unravel_index(max_idx, patch.shape) 
            # unravel_i converts it to (row, col)
            
            dX[h_start + max_pos[0], w_start + max_pos[1]] += dout[i, j]
            # entire backward logic
            # take the gradient form output dout[i, j]
            # assign to only max location
            # everything else = 0
            # += patches overlap, gradients accumulate
            
    return dX

# Average Pool

def avgpool_forward(X, k, stride):
    H, W = X.shape
    
    out_H = (H - k) // stride + 1
    out_W = (W - k) // stride + 1
    
    out = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + k

            w_start = j * stride
            w_end = w_start + k
            
            patch = X[h_start:h_end, w_start:w_end]
            out[i, j] = np.mean(patch)
            
    return out

def avgpool_backward(X, dout, k, stride):
    H, W, = X.shape
    dX = np.zeros_like(X) # ??
    
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


class CNNBlock:
    def __init__(self, kernel, pool_size=2, stride=2):
        self.kernel = kernel
        self.k = kernel.shape[0]
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, X):
        # input comes in
        self.X = X
        
        self.conv_out = conv2d_forward(X, self.kernel)
        # input is broken into patches
        # each patch is combined with kernel
        # u get a feature map
        
        self.relu_out = relu_forward(self.conv_out)
        
        self.pool_out = maxpool_forward(
            self.relu_out, self.pool_size, self.stride
        )
        # feature map is divided into patches
        # from each patch the strongest one is kept
        
        return self.pool_out
        # output that is passes into the next layer
    
    def backward(self, dout):
        dpool = maxpool_backward(
            self.relu_out, dout, self.pool_size, self.stride
        )
        # each pooled value came form a pax in patch
        # gradient only goes to max location
        # everything else gets 0 gradient
        
        drelu = relu_backward(self.conv_out, dpool)
        # input > 0: gradient passes
        # input <= 0: gradient beomes 0
        
        dX, dkernel = conv2d_backward(self.X, self.kernel, drelu)
        # does 2 things:
        # gradient wrt kernel: which patterns should the kernel learn more/less
        # gradient wrt input: how did each input pixel affect the loss
        
        self.dkernel = dkernel
        # optimizer will update using this gradient
        
        return dX
        # return input gradient to the prev layer
        