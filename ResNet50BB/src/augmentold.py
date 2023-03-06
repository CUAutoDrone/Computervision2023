import numpy as np
import cv2
from xml.etree import ElementTree as et
from config import IMAGE_EXTENSION

class DataAug:
  def __init__(self, augImgDir=None):
    self.saveImg = augImgDir != None
    self.augImgDir = augImgDir

  def __saveImg(self, img, tree, newImgName, xmin, ymin, xmax, ymax):
    if self.saveImg:

      tree = self.__editBBxml(tree, (xmin, ymin, xmax, ymax))
      name = newImgName.rpartition('/')[-1]
      cv2.imwrite(self.augImgDir + name + IMAGE_EXTENSION, img)
      tree.write(self.augImgDir + name + '.xml')

  def __getBB(self, tree):
    root = tree.getroot()

    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return (xmin, ymin, xmax, ymax)
      
  def __editBBxml(self, tree, boundingBoxCoords):
    root = tree.getroot()
    xmin, ymin, xmax, ymax = boundingBoxCoords
    root.find('object/bndbox/xmin').text = str(int(xmin))
    root.find('object/bndbox/ymin').text = str(int(ymin))
    root.find('object/bndbox/xmax').text = str(int(xmax))
    root.find('object/bndbox/ymax').text = str(int(ymax))

    return tree
  
  def __getImgInfo(self, imgName):
    img = cv2.imread(imgName + IMAGE_EXTENSION)
    h, w, _ = img.shape
    tree = et.parse(imgName + '.xml')
    xmin, ymin, xmax, ymax = self.__getBB(tree)
    return img, h, w, tree, xmin, ymin, xmax, ymax

  def __rotationMatrix(self, width,height,angle):
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

  def rotateImg(self, imgName, angle):
    img, h, w, tree, xmin, ymin, xmax, ymax = self.__getImgInfo(imgName)

    M = self.__rotationMatrix(w,h,angle)
    img = cv2.warpAffine(img, M, (w,h))

    boxEdges = np.array([xmin, ymax, xmax, ymax, xmin, ymin, xmax, ymin])
    boxEdges = boxEdges.reshape(-1,2)
    boxEdges = np.hstack((boxEdges, np.ones((boxEdges.shape[0],1), dtype = type(boxEdges[0][0]))))
    boxEdges = np.dot(M,boxEdges.T).T
    boxEdges = boxEdges.reshape(-1,2)
    xmax, ymax = np.max(boxEdges, axis=0)
    xmin, ymin = np.min(boxEdges, axis=0)

    self.__saveImg(img, tree, imgName+'rotated', xmin, ymin, xmax, ymax)
    return img, xmin, ymin, xmax, ymax

  def randomCrop(self, imgName):
    img, h, w, tree, xmin, ymin, xmax, ymax = self.__getImgInfo(imgName)

    xstart = np.random.randint(0, xmin + (xmax-xmin)//4)
    xstop = np.random.randint(xmax - (xmax-xmin)//4, w)
    ystart = np.random.randint(0, ymin + (ymax-ymin)//4)
    ystop = np.random.randint(ymax - (ymax-ymin)//4, h)

    img = cv2.resize(img[ystart:ystop, xstart:xstop], (w,h), interpolation=cv2.INTER_CUBIC)

    xmin = 0 if xstart >= xmin else w * (xmin - xstart)/(xstop - xstart)
    ymin = 0 if ystart >= ymin else h * (ymin - ystart)/(ystop - ystart)
    xmax = w if xstop <= xmax else w - (w * (xstop - xmax)/(xstop - xstart))
    ymax = h if ystop <= ymax else h - (h * (ystop - ymax)/(ystop - ystart))

    self.__saveImg(img, tree, imgName+'cropped', xmin, ymin, xmax, ymax)
    return img, xmin, ymin, xmax, ymax

  def flip(self, imgName):
    img, _, w, tree, xmin, ymin, xmax, ymax = self.__getImgInfo(imgName)

    img = img[:,::-1]
    xmin, xmax = w-xmax, w-xmin

    self.__saveImg(img, tree, imgName+'flipped', xmin, ymin, xmax, ymax)
    return img, xmin, ymin, xmax, ymax

  def randomPerspective(self, imgName):
    img, h, w, tree, xmin, ymin, xmax, ymax = self.__getImgInfo(imgName)

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

    self.__saveImg(img, tree, imgName+'perspective', xmin, ymin, xmax, ymax)
    return img, xmin, ymin, xmax, ymax

  def nonGeometric(self, imgName, func, params):
    """
    Applys a non geometric image transformation to the given image name and returns the image and bounding box as a tuple: img, xmin, ymin, xmax, ymax
    """
    img, _, _, tree, xmin, ymin, xmax, ymax = self.__getImgInfo(imgName)
    
    img = func(img, *params)
    self.__saveImg(img, tree, imgName + func.name, xmin, ymin, xmax, ymax)
    return img, xmin, ymin, xmax, ymax


  def gaussBlur(self, img, strength):
    """
    Adds gaussian blur to the given image, strength is the size of the blur in pixels, recommended from [2, 10]
    """
    strength = strength if strength % 2 == 1 else strength + 1
    kernel = cv2.getGaussianKernel(strength, -1)
    return cv2.filter2D(cv2.filter2D(img, -1, kernel), -1, kernel.T)
  gaussBlur.name = 'gaussianblur'

  def motionBlur(self, img, strength, angle):
    """
    Adds motion blur to the given image, angle is in degrees from horizontal, strength is the length in pixels of the blur, recommended from [2, 10]
    """
    strength = max(1, strength)
    kernel = np.zeros((strength, strength), dtype=np.float32)
    kernel[ (strength-1)// 2 , :] = np.ones(strength, dtype=np.float32)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D( (strength / 2, strength / 2) , angle, 1.0), (strength, strength) )  
    kernel = kernel * ( 1.0 / np.sum(kernel) )
    return cv2.filter2D(img, -1, kernel) 
  motionBlur.name = 'motionblur'

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

  def addNoise(self, img, strength):
    """
    Adds noise to the given image, strength varies from [0, 1], 0 is very little noise, 1 is visible noise
    """
    strength = np.clip(np.interp(strength, [0,1], [1.5,10]), 1.5, 10)
    w,h,c = img.shape
    noisep = strength*np.random.rand(w,h,c)
    noisem = strength*np.random.rand(w,h,c)
    return np.clip(np.subtract(np.add(img, noisep),noisem), 0, 255)
  addNoise.name = 'noise'