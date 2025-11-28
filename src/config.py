WINDOW_SIZE = 20            # frames per prediction window
WINDOW_STRIDE = 2           # stride between windows during training
PREDICTION_STABILITY = 4    # consecutive confirmations before emitting a label
CONF_THRESHOLD = 0.6        # softmax minimum for accepting a prediction
MODEL_PATH = "models/classifier.pth"
DATA_DIR = "data/raw"
