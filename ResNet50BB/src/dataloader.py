import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as et
import torchvision.transforms as transforms
from config import TRAIN_DIR, MAX_AUGMENTATIONS, AUGMENT_SAVE_DIR, IMAGE_EXTENSION, RESIZE_TO_X, RESIZE_TO_Y, VALIDATION_DIR, BATCH_SIZE, CLASSES, AUGMENTED_DIR
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from augment import DataAug
from PIL import Image

def collate_fn(batch):
    return tuple(zip(*batch))

def getNames(dir):
    data_names = []
    for filename in os.listdir(dir):
        if filename.endswith('.png'):
            name_without_extension = os.path.splitext(filename)[0]
            data_names.append(dir + name_without_extension)

    return data_names

def getImgBB(data_names):
    data = []
    for filename in data_names:
        img = cv2.imread(filename + IMAGE_EXTENSION)
        tree = et.parse(filename + '.xml')
        root = tree.getroot()
        label = str(root.find('object/name').text)
        xmin = int(root.find('object/bndbox/xmin').text)
        ymin = int(root.find('object/bndbox/ymin').text)
        xmax = int(root.find('object/bndbox/xmax').text)
        ymax = int(root.find('object/bndbox/ymax').text)

        data.append([img, label, xmin, ymin, xmax, ymax])

    return data

def augmentData(data_names):
    data = []
    bimodal = lambda : 1 + (np.random.choice([-1, 1]) * np.random.uniform(0.25,1))
    choose = lambda lst, number : [lst[i] for i in np.random.permutation( range(len(lst)) )[:number]]

    AugmentationManager = DataAug(savedImgDir=AUGMENT_SAVE_DIR)
    augment = {
        AugmentationManager.rotateImg         : lambda : (np.random.choice([-1, 1]) * np.random.randint(5,25),),
        AugmentationManager.randomCrop        : lambda : (),
        AugmentationManager.flip              : lambda : (),
        AugmentationManager.randomPerspective : lambda : (),
        AugmentationManager.gaussianBlur      : lambda : (np.random.randint(2,10),),
        AugmentationManager.motionBlur        : lambda : (np.random.randint(3,10), np.random.randint(0,360)),
        AugmentationManager.modifyCSL         : lambda : (bimodal(), bimodal(), bimodal()),
        AugmentationManager.addNoise          : lambda : (np.random.randint(-25,25),)
    }

    # Create list of augmentation types, all images have 1 random augment, ~37% have 2, ~13% have 3, in general e^-(x-1)
    augmentationTypes = [ [ choose(list(augment), i+1) for _ in range(int(percentage*len(data_names))) ] for i, percentage in enumerate(np.exp(np.multiply(-1,range(MAX_AUGMENTATIONS)))) ]

    # For every augmentation sequence apply them to the data
    for numAugments in augmentationTypes:
        data_names = np.random.permutation(data_names)
        for imgName, aug in zip(data_names, numAugments):
            augmentedImage = AugmentationManager.applyAug(imgName, aug, [augment[l]() for l in aug])
            data.append(list(augmentedImage))
    return data


def regularizeData(data):
    for i in range(len(data)):
        img, label, xmin, ymin, xmax, ymax = data[i]
        w, h, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (RESIZE_TO_X, RESIZE_TO_Y))
        img /= 255.0

        xmin *= RESIZE_TO_X/w
        xmax *= RESIZE_TO_X/w
        ymin *= RESIZE_TO_Y/h
        ymax *= RESIZE_TO_Y/h

        data[i] = [img, label, xmin, ymin, xmax, ymax]
    
    return data


class ImageDataSet(Dataset):
    def __init__(self, width, height, classes, data):
        self.width = width
        self.height = height
        self.classes = classes
        self.data = data

    def __getitem__(self, idx):
        img, label, xmin, ymin, xmax, ymax = self.data[idx]

        target = {}
        target["boxes"] = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        target["labels"] = torch.as_tensor([self.classes.index(label)], dtype=torch.int64)
        target["area"] = torch.tensor([(xmax - xmin) * (ymax - ymin)])
        target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        transform = transforms.ToTensor()

        img = transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)



train_names = getNames(TRAIN_DIR)
valid_names = getNames(VALIDATION_DIR)

train_data = getImgBB(train_names)
valid_data = getImgBB(valid_names)


if AUGMENTED_DIR == None:
    for data in augmentData(train_names):
        train_data.append(data)
else:
    augment_names = getNames(AUGMENTED_DIR)
    for data in getImgBB(augment_names):
        train_data.append(data)

train_data = regularizeData(train_data)
valid_data = regularizeData(valid_data)

train_dataset = ImageDataSet(RESIZE_TO_X, RESIZE_TO_Y, CLASSES, train_data)
valid_dataset = ImageDataSet(RESIZE_TO_X, RESIZE_TO_Y, CLASSES, valid_data)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(valid_data)}\n")

if __name__ == '__main__':
    train_names = getNames(TRAIN_DIR)

    train_data = getImgBB(train_names)

    augmentData(train_names)
    


    file_out = './bbs/'
    if file_out == None:
        quit()
    for filename in os.listdir(AUGMENT_SAVE_DIR):
        if filename.endswith('.png'):
            name_without_extension = os.path.splitext(filename)[0]
            # Load the PNG image
            image = Image.open(AUGMENT_SAVE_DIR + name_without_extension + IMAGE_EXTENSION)

            # Load the XML file and extract the bounding box coordinates
            tree = et.parse(AUGMENT_SAVE_DIR + name_without_extension + '.xml')
            root = tree.getroot()
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)

            # Draw the bounding box on the image
            draw = ImageDraw.Draw(image)
            draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=3)

            # Show the image with the bounding box
            image.save(file_out + name_without_extension + IMAGE_EXTENSION)