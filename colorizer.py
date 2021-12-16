from colorizers import *
from helper import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import color
import sys

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_eccv16.cuda()

path = "./imgs/"+sys.argv[1]+".jpg"
(img, grayscale_img) = load_img(path)
plt.imsave("./grayscale/grayscale_"+sys.argv[1]+".jpg", grayscale_img)

(original_tensor, resized_tensor) = extract_l_channel(grayscale_img)
resized_tensor = resized_tensor.cuda()
result_lab_t = concat_l_ab_channels(original_tensor, colorizer_eccv16(resized_tensor).cpu())
img = color.rgb2lab(img)
img = color.lab2rgb(img)

plt.imsave("./results/colorized_"+sys.argv[1]+".jpg", result_lab_t)

print(pixelAccuracy(result_lab_t, img))
