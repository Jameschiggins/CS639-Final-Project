import numpy as np
from PIL import Image
from skimage import color
import torch
import torch.nn.functional as F

def load_img(path):
    img = np.asarray(Image.open(path))
    if(img.ndim == 2):
        img = np.tile(img[:, :, None], 3)
    return img

def extract_l_channel(img):
    # Grab L channel of original image
    img_lab = color.rgb2lab(img)
    img_l_channel = img_lab[:,:,0]

    # Resize original image and grab L channel of resized image
    resized_rgb = np.asarray(Image.fromarray(img).resize((256, 256)), 3)
    resized_lab = color.rgb2lab(resized_rgb)
    resized_l_channel = resized_lab[:,:,0]

    original_tensor = torch.Tensor(img_l_channel)[None,None,:,:]
    resized_tensor = torch.Tensor(resized_l_channel)[None,None,:,:]
    return (original_tensor, resized_tensor)

def concat_l_ab_channels(original_tensor, ab):
    img_orig = original_tensor.shape[2:] # One hot tensor that is 1x[1xheightxwidth]
    img_ab = ab.shape[2:] # One hot tensor that is 1x[1x2xheightxwidth] L & AB with image

    if(img_orig[0]!=img_ab[0] or img_orig[1]!=img_ab[1]):
        img_ab = F.interpolate(ab, size=img_orig, mode='bilinear')

    result_lab = torch.cat((img_orig, img_ab), dim=1)
    result_lab_t = result_lab.cpu().numpy()[0,...].transpose((1,2,0))
    return color.lab2rgb(result_lab_t)
