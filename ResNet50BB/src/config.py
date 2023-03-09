import torch

NUM_EPOCHS = 100 # Number of epochs to train for
BATCH_SIZE = 4 # Mini batch size

# Size of image for model input
RESIZE_TO_X = 360 
RESIZE_TO_Y = 512

# Image and XML source directories
TRAIN_DIR = './data/train/' 
VALIDATION_DIR = './data/test/'

AUGMENT_DIR = './out/aug/' # Where to save augmented images, None if unsaved

OUT_DIR = './out/' # Output directory 
IMAGE_EXTENSION = '.png' # Image type

MAX_AUGMENTATIONS = 4 # Max number of augmentations applied to any image


# Classes to categorize, 0 index reserved for background
CLASSES = [
    'background',
    'mast'
]
NUM_CLASSES = len(CLASSES)

SAVE_MODEL_EPOCH = 2 # How often to save model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')