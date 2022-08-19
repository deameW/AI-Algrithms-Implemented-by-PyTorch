import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

m1 = nn.Conv2d(16,33,3,stride=2) # input_channel, output_channel, kernal_size,stride
# 定义输入
input = torch.randn(20,16,50,100) # batch, channel, height, width

output=m1(input)
print(output.detach().numpy().shape)

"""
    特征图尺寸：
        (H/W - kernel_size + padding*2) / stride
"""



image = Image.open('../data/edge_deection.jpg') # 300x300
# 指定色彩模式，L: 8位像素，黑白
image = image.convert("L")
image_np = np.array(image)
image = image.convert("L")

# 创建tensor
h,w = image_np.shape
image_tensor = torch.from_numpy(image_np.reshape(1, 1, h, w)).float()

# 卷积核
kersize = 3
kernel = torch.tensor([[1.,1,1],[1,1,1],[1,1,1]])

conv2d = torch.nn.Conv2d(1, 1, (kersize, kersize), bias=False)
kernel = kernel.reshape((1, 1, kersize, kersize))
conv2d.weight.data = kernel

image_out = conv2d(image_tensor)
image_out = image_out.data.squeeze()
plt.axis('off')
plt.imshow(image_out, cmap=plt.cm.gray)
plt.show()