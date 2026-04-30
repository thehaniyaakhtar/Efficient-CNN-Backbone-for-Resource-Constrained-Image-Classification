# Architecture Power

# Residual Connections
class ResidualBlock:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        
    def forward(self, X):
        self.X = X
        
        self.out1 = conv2d_forward(X, self.k1)
        self.relu1 = relu_forward(self.out1)
        
        self.out2 = conv2d_forward(self.relu1, self.k2)
        
        self.out = self.out2 + X
        return self.out

'''
output = processed input + original input
with residual, input goes through layers
original input is still kept instead of having it changed
final output is a mix of both

this helps as:
layers mess things up/
change too much
this can lead to info getting lost/ distorted

residual connection ensures that original input stays intact
final output becomes:
F(X) + X

the network does not try to learn the whole thing from scratch,
it learns corrctions/adjustments to the org input
'''

# Depthwise Seperable Convolution
import numpy as np
def depthwise_conv(X, kernels):
    outputs = []
    for c in range(len(kernels)):
        outputs.append(conv2d_forward(x[c], kernels[c]))
        # each channel is processed independently
    return np.array(outputs)

def pointwise_conv(X, weights):
    F, C = weights.shape
    H, W = X.shape[1:]
    
    out = np.zeros((F, H, W))
    for f in range(F):
        for c in range(C):
            out[f] += weights[f, c] * X[c]
            # combine channels using weights
    return out
    
'''
In normal distribution, one filter looks at all input channels at once and mixes everything in a single step
this makes it slow and computationally expensive

Step 1: learning paterns inside each channel
No mixing channel, we process each channel seperately 
3 channels (R, G, B), 3 seperate filters
each filter has its own channel

Step 2: Pointwise Convolution
use 1x1 filters, that look across channels and mix
features from different channels interact

depthwise: spatial filtering
pointwise: channel mixing
'''

# MobileNet Block

class MobileNetBlock:
    def __init__(self, expand_w, depth_k, project_w):
        self.expand_x = expand_w
        self.depth_k = depth_k
        self.project_w = project_w
        
    def forward(self, X):
        self.X = X
        
        self.expand = pointwise_conv(X, self.expand_w)
        
        self.depth = depthwise_conv(self.expanded, self.depth_k)
        
        self.projected = pointwise_conv(self.depth, self.project_w)
        
        return self.projected
    
'''
efficient version of a convolution block
step 1: input is expanded to more channels using 1x1 conv
taking smaller number of channels, inc them to a larger number 
gives more capacity to learn features

Step2: depthwise conv; leanrning spatials patterns(edges, textures)
each channel is procesed independently 
no mixing between channels
each channel is processes independently
no mixing between channels

Step3: Project (Compress)
reduce the numver of channels back using 1x1 conv
combines info across channels
reduces size
'''

# Squeeze and Excitation (SE Block)
def se_block(X):
    
    # sqeeze
    z = np.mean(X, axis = (1, 2))
    
    # excitation
    scale = 1 / (1 + np.exp(-z))
    # sigmoid
    
    # reweight
    out = X * scale[:, None, None]
    
    return out

'''
Helps decide which channels are imp and which are not
each channel represents a diff type of feature
the SE block lets the model focus on which are imp and reduce the 
effct of less useful ones

Step 1: Squeeze 
Compress each channel into a single number, by taking the avg of all values in the channel
each channel thus has one value representing its imp

Step2: Excitation
Convert those values between 0 and 1

Step3: Reweight
Scale each channel using its weight
imp chanels: multiplied by values close to 1: stay strong
less imp: mult by values close to 0: get suppressed
'''



