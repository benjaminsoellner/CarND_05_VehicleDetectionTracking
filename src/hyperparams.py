"""
Contains the hyperparameters for the algorithms 
"""

HYPERPARAMS = {
    'COLOR_SPACE': 'YUV', # Color space
    'SPATIAL_FEAT': False, # Using spacial features?
    'SPATIAL_SIZE': (32, 32), # How many spacial features
    'HOG_FEAT': True, # Using hog features?
    'HOG_ORIENT': 11, # Number of possible hog cell orientations
    'HOG_PIX_PER_CELL': 8, # Pixels per hog cell
    'HOG_CELL_PER_BLOCK': 2, # Hog cell overlap
    'HOG_CHANNEL': 'ALL', # Which hog channels to use
    'HOG_SQRT': False, # use hog's sqrt_transform=True parameter?
    'HIST_FEAT': True, # Using histogram features?
    'HIST_RANGE': (0.0, 1.0), # Histogram min / max
    'HIST_BIN': 32, # How many bins for histogram
    'Y_START': 350, # Top pixel
    'Y_STOP': 656, # Bottom pixel
    'RESCALES': [1.0], # Scales of windows based on 64x64 window being scale 1.0
    'HEAT_THRESHOLD': 5.0, # Threshold to reject false positives
    'HEAT_FRAMES': 40 # Over how many frames to collect heat
}


