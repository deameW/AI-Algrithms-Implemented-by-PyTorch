from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

if __name__ == "__main__":
    image = Image.open('../data/edge_deection.jpg') # 300x300
    # 指定色彩模式，L: 8位像素，黑白
    # image = image.convert("L")
    image_np = np.array(image)

    # 创建tensor
    h,w = image_np.shape
    image_tensor = torch.from_numpy(image_np.reshape(1, 1, h, w)).float()

    # 卷积核
    kersize = 5
    kernel = torch.ones(kersize,kersize,dtype=torch.float32) * -1

    conv2d = torch.nn.Conv2d(1,1,(kersize,kersize),bias=False)
    kernel = kernel.reshape((1,1,kersize,kersize))
    conv2d.weight.data = kernel

    image_out = conv2d(image_tensor)
    image_out = image_out.data.squeeze()
    plt.axis('off')
    plt.imshow(image_out, cmap=plt.cm.gray)
    plt.show()