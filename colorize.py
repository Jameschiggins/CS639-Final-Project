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
    img_lab = color.rgb2lab(img)
    img_l_channel = img_lab[:,:,0]

    # Resize original image and grab L channel of resized image
    resized_rgb = np.asarray(Image.fromarray(img).resize((256, 256)))
    resized_lab = color.rgb2lab(resized_rgb)
    resized_l_channel = resized_lab[:,:,0]

    original_tensor = torch.Tensor(img_l_channel)[None,None,:,:]
    resized_tensor = torch.Tensor(resized_l_channel)[None,None,:,:]
    return (original_tensor, resized_tensor)
