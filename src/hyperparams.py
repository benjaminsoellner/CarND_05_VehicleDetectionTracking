

# Hyperparameters


HYPERPARAMS = {
    'COLOR_SPACE': 'RGB', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'SPATIAL_FEAT': False,
    'SPATIAL_SIZE': (32, 32),
    'HOG_FEAT': True,
    'HOG_ORIENT': 9,
    'HOG_PIX_PER_CELL': 8,
    'HOG_CELL_PER_BLOCK': 2,
    'HOG_CHANNEL': 'ALL',  # Can be 0, 1, 2, or "ALL"
    'HIST_FEAT': True,
    'HIST_RANGE': (0.0, 1.0),
    'HIST_BIN': 32,
    'Y_START': 400,
    'Y_STOP': 656,
    'RESCALES': [1.5, 3.0, 6.0]
}


