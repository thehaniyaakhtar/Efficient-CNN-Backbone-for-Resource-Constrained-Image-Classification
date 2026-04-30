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