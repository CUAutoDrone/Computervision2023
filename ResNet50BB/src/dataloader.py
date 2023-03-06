import os
import numpy as np
from augment import DataAug
from PIL import Image, ImageDraw
from config import TRAIN_DIR, MAX_AUGMENTATIONS
import xml.etree.ElementTree as ET

choose = lambda lst, number : [lst[i] for i in np.random.permutation( range(len(lst)) )[:number]]
bimodal = lambda : 1 + (np.random.choice([-1, 1]) * np.random.uniform(0.25,1))

savedImgDir='./out/'

AugmentationManager = DataAug(savedImgDir=savedImgDir)

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

# Load data from TRAIN_DIR
data = []
for filename in os.listdir(TRAIN_DIR):
    if filename.endswith(".png"):
        name_without_extension = os.path.splitext(filename)[0]
        data.append(name_without_extension)


# Create list of augmentation types, all images have 1 random augment, ~37% have 2, ~13% have 3 etc.
augmentationTypes = [ [ choose(list(augment), i+1) for _ in range(int(percentage*len(data))) ] for i, percentage in enumerate(np.exp(np.multiply(-1,range(MAX_AUGMENTATIONS)))) ]

# For every augmentation sequence apply them to the data
for numAugments in augmentationTypes:
    data = np.random.permutation(data)
    for imgName, aug in zip(data, numAugments):
        AugmentationManager.applyAug(TRAIN_DIR + imgName, aug, [augment[l]() for l in aug], saveImg=True)





# Take all augmented images and draw bounding boxes, save to file_out

file_out = "./bbs/"


for filename in os.listdir(savedImgDir):
    if filename.endswith(".png"):
        name_without_extension = os.path.splitext(filename)[0]
        # Load the PNG image
        image = Image.open(savedImgDir+name_without_extension+'.png')

        # Load the XML file and extract the bounding box coordinates
        tree = ET.parse(savedImgDir+name_without_extension+'.xml')
        root = tree.getroot()
        xmin = int(root.find('object/bndbox/xmin').text)
        ymin = int(root.find('object/bndbox/ymin').text)
        xmax = int(root.find('object/bndbox/xmax').text)
        ymax = int(root.find('object/bndbox/ymax').text)

        # Draw the bounding box on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=3)

        # Show the image with the bounding box
        image.save(file_out+name_without_extension+".png")
