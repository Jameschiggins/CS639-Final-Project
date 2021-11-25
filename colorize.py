import numpy as np
from PIL import Image

def load_img(path):
    img = np.asarray(Image.open(path))
    if(img.ndim == 2):
        img = np.tile(img[:, :, None], 3)
    return img