# ViT Patch Embedding

import numpy as np

def patch_embedding(X, patch_size):
    H, W = X.shape
    patches = []
    
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # loop over patches, splitting imgs into blocks
            patch = X[i:i+patch_size, j:j+patch_size]
            # extract patches
            patch.flatten()
            # faltten
            patches.append(patch.flatten())
            # store as tokens
    
    return np.array(patches)

'''
Patch embeddings is the first step in turning an img into a Transformer can understand
They dont work directly on images
They work on sequences
Convert image into a sequences

# Step1: Split image into patches 
Breaking the image into small square patches
getting mult small patches

# Step2: Flatten each patch
Each patch is still 2D
Convert to 1D vector
(7x7) -> (49,)

# Step3: Treat each patch as a token
Now each flattened patch becomes:

'''