import numpy as np
from PIL import Image
from skimage import color
import torch
import torch.nn.functional as F
from colorizers import *

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
colorizer_eccv16.cuda()
colorizer_siggraph17.cuda()

def load_img(path):
    img = np.asarray(Image.open(path))
    if(img.ndim == 2):
        img = np.tile(img[:, :, None], 3)
    return img

def extractLChannel(img):
    # Grab L channel of original image
    img_labColor = color.rgb2lab(img)
    img_lChannel = img_labColor[:,:,0]

    # Resize original image and grab L channel of resized image
    resized_rgb = np.asarray(Image.fromarray(img).resize((256, 256)))
    resized_labColor = color.rgb2lab(resized_rgb)
    resized_lChannel = resized_labColor[:,:,0]

    originalTensor = torch.Tensor(img_lChannel)[None,None,:,:]
    resizedTensor = torch.Tensor(resized_lChannel)[None,None,:,:]
    return (originalTensor, resizedTensor)
