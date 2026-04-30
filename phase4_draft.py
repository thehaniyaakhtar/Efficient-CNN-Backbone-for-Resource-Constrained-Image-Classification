# Advanced Features

import numpy as np

# Global Average Pooling
def global_avg_pool(X):
    return np.mean(X, axis=(1, 2))


# Spatial Pyramid Pooling (SPP)
def spp(X):
    p1 = np.max(X) # one value of whole img
    p2 = maxpool_forward(X, 2, 2).flatten() # 2x2 regions
    p3 = maxpool_forward(X, 4, 4).flatten() # smaller regions
    
    return np.concatenate(([p1], p2, p3)) # combined

'''
Used to capture info from an img at diff scales
instead of looking at the img in one way, it:
looks at the whole image
             medium regions
             small regions
             
it combines all the info together

in normal pooling the image is reduced once
u might lose imp details

applying pooling:
at global level:
    look at the entire img, gives one value
at medium level:
    split image into a few regions -> pool each 
at final level:
    split into more regions -> pool each 
    
Then everything is combined, final output is a mix of
global info
medium info
fine details

'''
