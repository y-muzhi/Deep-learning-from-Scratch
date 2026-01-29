# ch03/mnist_show.py
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(base_dir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 例如：5

print(img.shape)           # (784,)
img = img.reshape(28, 28)  # 还原为 28x28
print(img.shape)           # (28, 28)

img_show(img)
