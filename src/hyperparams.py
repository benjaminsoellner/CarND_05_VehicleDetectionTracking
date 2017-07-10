

# Hyperparameters


HYPERPARAMS = {
    'COLOR_SPACE': 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'SPATIAL_FEAT': False,
    'SPATIAL_SIZE': (32, 32),
    'HOG_FEAT': True,
    'HOG_ORIENT': 11,
    'HOG_PIX_PER_CELL': 8,
    'HOG_CELL_PER_BLOCK': 2,
    'HOG_CHANNEL': 'ALL',  # Can be 0, 1, 2, or "ALL"
    'HOG_SQRT': False,
    'HIST_FEAT': True,
    'HIST_RANGE': (0.0, 1.0),
    'HIST_BIN': 32,
    'Y_START': 350,
    'Y_STOP': 656,
    'RESCALES': [1.0],
    'HEAT_THRESHOLD': 15.0,
    'HEAT_FRAMES': 30
}


