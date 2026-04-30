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

# Positional Encoding

def positional_encoding(seq_len, dim):
    pos = np.arange(seq_len)[:, None]
    # gives position index (0, 1, 2, ...)
    i = np.arange(dim)[None, :]
    # Dimension index
    
    angle = pos / (10000 ** (2*(i//2)/dim))
    # create angle values
    pe = np.zeros((seq_len, dim))
    
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    # Applying sin and cos to create unique patter per position
    
    return pe

'''
The image is then converted into a sequence of tokens
the model now onli sees list of tokens, dk where the patch comes from
it loses spatial info like 
top vs bottom
left vs right

Thus positional encoding hels the model know 
the location of the patch

It creates a vector for each position in the sequece
each pattern gets a unique pattern of numbers
pattern generated using:
sine
cosine

for each token:
final token = patch_embedding + positional_encoding
'''

# Local Response Normalization

def lrn(X, k=2, alpha=1e-4, beta=0.75):
    sq = X**2
    # square activations, measures activation strength
    scale = k + alpha * np.sum(sq, axis=0)
    # np.sum...: check total channel activity at each spatial position
    # larger activity, larger scale
    return X / (scale ** beta)
    # Strong competition reduces value

'''
Used to make channels complete with each other 
diff channels select diff features
many channels can become strongly sctivated at the same location
this makes the model give too much imp to several features at once
LRN fixes this by reducing values when nearby channels are also large
this encourages strongest features stand out
instead of many channels activating equally
'''
