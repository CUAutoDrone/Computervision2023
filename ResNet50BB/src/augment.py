import numpy as np
import cv2
from xml.etree import ElementTree as et
from config import IMAGE_EXTENSION

class DataAug:
  def __init__(self, savedImgDir=None):
    self.savedImgDir = savedImgDir

  def __saveImg(self, img, tree, newImgName, xmin, ymin, xmax, ymax):
    """
    Saves the given image and modified XML tree (to have given bounding box) to the savedImgDir with the name given by newImgName
    """
    tree = self.__editBBxml(tree, (xmin, ymin, xmax, ymax))
    name = newImgName.rpartition('/')[-1]
    cv2.imwrite(self.savedImgDir + name + IMAGE_EXTENSION, img)
    tree.write(self.savedImgDir + name + '.xml')

  def __getBB(self, tree):
    """
    Given an XML tree, returns the bounding box coordinates in the form: xmin, ymin, xmax, ymax
    """
    root = tree.getroot()

    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return xmin, ymin, xmax, ymax
      
  def __editBBxml(self, tree, boundingBoxCoords):
    """
    Given an XML tree, returns a new tree identical except with new bounding box coordinates, boundingBoxCoords takes a tuple of the form: xmin, ymin, xmax, ymax
    """
    root = tree.getroot()
    xmin, ymin, xmax, ymax = boundingBoxCoords
    root.find('object/bndbox/xmin').text = str(int(xmin))
    root.find('object/bndbox/ymin').text = str(int(ymin))
    root.find('object/bndbox/xmax').text = str(int(xmax))
    root.find('object/bndbox/ymax').text = str(int(ymax))

    return tree

  def __rotationMatrix(self, width, height, angle):
    s = height/width if height > width else width/height
    if angle == 90:
      return cv2.getRotationMatrix2D((width/2,height/2),90,s)
    if angle == -90:
      return cv2.getRotationMatrix2D((width/2,height/2),-90,s)

    if (angle > 0):
      phi = np.arctan(width/height)
    else:
      phi = np.arctan(width/height)
    theta = 2*np.pi - np.abs(np.deg2rad(angle)) - phi
    scale = np.abs(np.sin(theta)/np.sin(phi))
    return cv2.getRotationMatrix2D((width/2,height/2),angle,scale)

  def rotateImg(self, img, xmin, ymin, xmax, ymax, angle):
    """
    Rotates the given image by an amount specified in degrees, scales the image to prevent extra space, preserves bounding box
    """
    h, w, _ = img.shape

    M = self.__rotationMatrix(w,h,angle)
    img = cv2.warpAffine(img, M, (w,h))

    boxEdges = np.array([xmin, ymax, xmax, ymax, xmin, ymin, xmax, ymin])
    boxEdges = boxEdges.reshape(-1,2)
    boxEdges = np.hstack((boxEdges, np.ones((boxEdges.shape[0],1), dtype = type(boxEdges[0][0]))))
    boxEdges = np.dot(M,boxEdges.T).T
    boxEdges = boxEdges.reshape(-1,2)

    xmax, ymax = np.max(boxEdges, axis=0)
    xmin, ymin = np.min(boxEdges, axis=0)

    xmin, xmax = np.clip((xmin, xmax), 0, w)
    ymin, ymax = np.clip((ymin, ymax), 0, h)

    return img, int(xmin), int(ymin), int(xmax), int(ymax)
  rotateImg.name = 'rotated'
  rotateImg.isGeometric = True

  def randomCrop(self, img, xmin, ymin, xmax, ymax):
    """
    Randomly crops the given image, possible stretching the image, preserves bounding box
    """
    h, w, _ =  img.shape

    xstart = np.random.randint(0, xmin + (xmax-xmin)//4)
    xstop = np.random.randint(xmax - (xmax-xmin)//4, w)
    ystart = np.random.randint(0, ymin + (ymax-ymin)//4)
    ystop = np.random.randint(ymax - (ymax-ymin)//4, h)

    img = cv2.resize(img[ystart:ystop, xstart:xstop], (w,h), interpolation=cv2.INTER_CUBIC)

    xmin = 0 if xstart >= xmin else w * (xmin - xstart)/(xstop - xstart)
    ymin = 0 if ystart >= ymin else h * (ymin - ystart)/(ystop - ystart)
    xmax = w if xstop <= xmax else w - (w * (xstop - xmax)/(xstop - xstart))
    ymax = h if ystop <= ymax else h - (h * (ystop - ymax)/(ystop - ystart))

    return img, int(xmin), int(ymin), int(xmax), int(ymax)
  randomCrop.name = 'cropped'
  randomCrop.isGeometric = True

  def flip(self, img, xmin, ymin, xmax, ymax):
    """
    Horizontally flips the given image, preserves the bounding box
    """
    _, w, _ = img.shape

    img = img[:,::-1]
    xmin, xmax = w-xmax, w-xmin

    return img, int(xmin), int(ymin), int(xmax), int(ymax)
  flip.name = 'flipped'
  flip.isGeometric = True

  def randomPerspective(self, img, xmin, ymin, xmax, ymax):
    """
    Performs a random perspective shift on the given image, preserves the bounding box
    """
    h, w, _ = img.shape

    x1, x3 = np.random.randint(0, xmin + (xmax-xmin)//4, size=2)
    x2, x4 = np.random.randint(xmax - (xmax-xmin)//4, w, size=2)
    y3, y4 = np.random.randint(0, ymin + (ymax-ymin)//4, size=2)
    y1, y2 = np.random.randint(ymax - (ymax-ymin)//4, h, size=2)

    pts1 = np.float32([[x1, y1], [x2, y2],[x3, y3], [x4, y4]])
    pts2 = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
    
    mask = np.zeros((h,w))
    mask = cv2.rectangle(mask, (xmin,ymin), (xmax,ymax), (255,255,255), -1)

    M = cv2.getPerspectiveTransform(pts1, pts2)                       
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_CUBIC)

    x = np.where(np.max(mask, axis=0) > 200)
    y = np.where(np.max(mask, axis=1) > 200)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    return img, int(xmin), int(ymin), int(xmax), int(ymax)
  randomPerspective.name = 'perspective'
  randomPerspective.isGeometric = True

  def gaussianBlur(self, img, strength):
    """
    Adds gaussian blur to the given image, strength is the size of the blur in pixels, recommended from [2, 10]
    """
    strength = strength if strength % 2 == 1 else strength + 1
    kernel = cv2.getGaussianKernel(strength, -1)
    return cv2.filter2D(cv2.filter2D(img, -1, kernel), -1, kernel.T)
  gaussianBlur.name = 'gaussianblur'
  gaussianBlur.isGeometric = False

  def motionBlur(self, img, strength, angle):
    """
    Adds motion blur to the given image, angle is in degrees from horizontal, strength is the length in pixels of the blur, recommended from [3, 10]
    """
    strength = max(3, strength)
    kernel = np.zeros((strength, strength), dtype=np.float32)
    kernel[ (strength-1)// 2 , :] = np.ones(strength, dtype=np.float32)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D( (strength / 2, strength / 2) , angle, 1.0), (strength, strength) )  
    kernel = kernel * ( 1.0 / np.sum(kernel) )
    
    return cv2.filter2D(img, -1, kernel)
  motionBlur.name = 'motionblur'
  motionBlur.isGeometric = False

  def modifyCSL(self, img, contrast, saturation, luminance):
    """
    Changes the contrast saturation and luminance of an image, each value varies from [0, 2]. 1 is no change.
    """
    contrast = np.clip(np.interp(contrast, [0,2], [.5,1.5]), .5, 1.5)
    saturation = np.clip(np.interp(saturation, [0,2], [.5,1.5]), .5, 1.5)
    luminance = np.clip(np.interp(luminance, [0,2], [-25, 25]),-25, 25)

    luminance += int(round(255*(1-contrast)/2))
    img = cv2.addWeighted(img, contrast, img, 0, luminance)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    H, S, L = cv2.split(img)
    S = np.clip(S*saturation,0,255)
    img = cv2.merge([H,S,L])

    return cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)
  modifyCSL.name = 'colordistort'
  modifyCSL.isGeometric = False

  def addNoise(self, img, strength):
    """
    Adds noise to the given image, strength varies from [0, 1], 0 is very little noise, 1 is visible noise
    """
    strength = np.clip(np.interp(strength, [0,1], [1.5,10]), 1.5, 10)
    w,h,c = img.shape
    noisep = strength*np.random.rand(w,h,c)
    noisem = strength*np.random.rand(w,h,c)
    return np.clip(np.subtract(np.add(img, noisep),noisem), 0, 255).astype(np.uint8)
  addNoise.name = 'noise'
  addNoise.isGeometric = False

  def applyAug(self, imgName, funcs, params, saveImg=False):
    """
    Applys the specified image transformation(s) to the given file and returns the image and bounding box as a tuple: img, xmin, ymin, xmax, ymax

    imgName: the filename of the img and .xml file without the extension
    funcs: data augmentation function or iterable of functions to apply
    params: parameters for function or iterable of parameters Parameters must be in tuple, ie. () or (1,)
    saveImg: whether the resulting augmented image is saved to the savedImgDir along with modified xml file
    """
    img = cv2.imread(imgName + IMAGE_EXTENSION)
    tree = et.parse(imgName + '.xml')
    xmin, ymin, xmax, ymax = self.__getBB(tree)

    mods = ''

    try:
      iter(funcs)
    except:
      funcs = [funcs]
      params = [params]

    for func, param in zip(funcs, params):
      if func.isGeometric:
        img, xmin, ymin, xmax, ymax = func(img, xmin, ymin, xmax, ymax, *param)
      else:
        img = func(img, *param)
      mods += func.name
        
    if saveImg:
        self.__saveImg(img, tree, imgName + mods, xmin, ymin, xmax, ymax)
    
    return img, xmin, ymin, xmax, ymax
