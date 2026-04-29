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