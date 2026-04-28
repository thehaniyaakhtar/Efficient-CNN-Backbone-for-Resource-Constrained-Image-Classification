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