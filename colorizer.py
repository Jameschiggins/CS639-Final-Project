from colorizers import *
from helper import *
import matplotlib.pyplot as plt

colorizer_eccv16 = eccv16(pretrained=True).eval()
#colorizer_eccv16.cuda()

path = "imgs/ansel_adams.jpg"
img = load_img(path)

(original_tensor, resized_tensor) = extract_l_channel(img)
#resized_tensor = resized_tensor.cuda()
result_lab_t = concat_l_ab_channels(original_tensor, colorizer_eccv16(resized_tensor).cpu())

plt.imsave("colorized_ansel_adams.jpg", result_lab_t)
plt.imshow(result_lab_t)