from colorizers import *
from helper import *
import matplotlib.pyplot as plt
import numpy as np
import torch

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_eccv16.cuda()

path = "imgs/dog1.png"
img = load_img(path)

(original_tensor, resized_tensor) = extract_l_channel(img)
resized_tensor = resized_tensor.cuda()
result_lab_t = concat_l_ab_channels(original_tensor, colorizer_eccv16(resized_tensor).cpu())

plt.imsave("colorized_dog1_2.png", result_lab_t)
plt.imshow(result_lab_t)

print(pixelAccuracy(result_lab_t, img))

# imgColor = torch.from_numpy(result_lab_t)
# imgGround = torch.from_numpy(img)

# imgPred = np.asarray(torch.squeeze(imgColor))
# imgLab = np.asarray(torch.squeeze(imgGround))

# accuracy = np.empty(imgLab.shape[0])
# correct = np.empty(imgLab.shape[0])
# labeled = np.empty(imgLab.shape[0])

# for i in range(imgLab.shape[0]):
#     accuracy[i], correct[i], labeled[i] = pixelAccuracy(imgColor[i], imgGround[i])

# acc = 100.0 * np.sum(correct) / (np.spacing(1) + np.sum(labeled))
# print(acc)
