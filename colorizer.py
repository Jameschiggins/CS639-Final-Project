from colorizers import *
from helper import *
import matplotlib.pyplot as plt

colorizer_eccv16 = eccv16(pretrained=True).eval()

path = "imgs/grayscale_test1.jpg"
img = load_img(path)

(original_tensor, resized_tensor) = extract_l_channel(img)
result_lab_t = concat_l_ab_channels(original_tensor, colorizer_eccv16(resized_tensor).cpu())

plt.imsave("colorized_ansel_adams.jpg", result_lab_t)
plt.imshow(result_lab_t)