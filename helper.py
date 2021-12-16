import numpy as np
from PIL import Image
from skimage import color
import torch
import torch.nn.functional as F

def load_img(path):
    img = np.asarray(Image.open(path))
    grayscale_img = np.asarray(Image.open(path).convert('L'))
    if(img.ndim == 2):
        img = np.tile(img[:, :, None], 3)
    if(grayscale_img.ndim == 2):
        grayscale_img = np.tile(grayscale_img[:, :, None], 3)
    return (img, grayscale_img)

def extract_l_channel(img):
    # Grab L channel of original image
    img_lab = color.rgb2lab(img)
    img_l_channel = img_lab[:,:,0]

    # Resize original image and grab L channel of resized image
    resized_rgb = np.asarray(Image.fromarray(img).resize((256, 256), 3))
    resized_lab = color.rgb2lab(resized_rgb)
    resized_l_channel = resized_lab[:,:,0]

    original_tensor = torch.Tensor(img_l_channel)[None,None,:,:]
    resized_tensor = torch.Tensor(resized_l_channel)[None,None,:,:]
    return (original_tensor, resized_tensor)

def concat_l_ab_channels(original_tensor, ab):
    img_orig = original_tensor.shape[2:] # One hot tensor that is 1x[1xheightxwidth]
    img_ab = ab.shape[2:] # One hot tensor that is 1x[1x2xheightxwidth] L & AB with image

    if(img_orig[0]!=img_ab[0] or img_orig[1]!=img_ab[1]):
        result_ab = F.interpolate(ab, size=img_orig, mode='bilinear')
    else:
        result_ab = ab

    result_lab = torch.cat((original_tensor, result_ab), dim=1)
    result_lab_t = result_lab.data.cpu().numpy()[0,...].transpose((1,2,0))
    return color.lab2rgb(result_lab_t)

def pixelAccuracy(imgColor, imgGround):
    correct = 0
    for i in range(np.shape(imgGround)[0]):
        for j in range (np.shape(imgGround)[1]):
            for k in range (np.shape(imgGround)[2]):
                if abs(imgGround[i,j,k] - imgColor[i,j,k]) < 0.05:
                    correct = correct + 1

    accuracy = correct / (np.shape(imgGround)[2] * np.shape(imgGround)[1] * np.shape(imgGround)[0])

    return accuracy
