import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
DATA_ROOT    = "./data"
IMAGE_SIZE   = 128
NUM_CLASSES  = 3
VAL_SPLIT    = 0.1
RANDOM_SEED  = 42

# training
BATCH_SIZE   = 16
NUM_WORKERS  = 4
EPOCHS       = 25
LR           = 0.001

# scheduler
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA     = 0.1

# output
CHECKPOINT_PATH = "artifacts/unet.pth"
