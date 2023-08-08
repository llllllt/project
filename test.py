from utils import random_mask_single
import torchvision.transforms as transforms
from PIL import Image
import torch

# open a image
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为torch tensor
    # transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    transforms.Resize((256, 256), antialias=True) # resize
])
image = Image.open("astronaut_rides_horse.png")          

# convert to tensor
image_tensor = transform(image)
print(image_tensor.shape)

# mask image
masked_image = random_mask_single(image_tensor)
pil_image = transforms.ToPILImage()(masked_image)

# save masked image
output_path = "output_image.png"
pil_image.save(output_path)

