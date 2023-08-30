from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为torch tensor
    # transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    transforms.Resize((256, 256), antialias=True) # resize
])

class CustomDataset(Dataset):
    def __init__(self, fake_data_dir, real_data_dir, transform=transforms) -> None:
        super().__init__()
        self.fake_data_dir = fake_data_dir
        self.real_data_dir = real_data_dir
        self.transform = transform
        self.fake_image_paths = [file for file in os.listdir(fake_data_dir) if file.endswith('.jpg')]
        self.real_image_paths = [file for file in os.listdir(real_data_dir) if file.endswith('.png')]
        
    
    def __len__(self):
        return len(self.real_image_paths)
    
    def __getitem__(self, index):
        # print(index)
        # print(self.real_image_paths[index].split('.'))
        real_img_name = self.real_image_paths[index].split('.')[0]
        fake_img_name = real_img_name
        # print(real_img_name)
        real_image = Image.open(os.path.join(self.real_data_dir, real_img_name + '.png')).convert("RGB")
        fake_image = Image.open(os.path.join(self.fake_data_dir, fake_img_name + '.jpg')).convert("RGB")
        if self.transform:
            real_image = self.transform(real_image)
            fake_image = self.transform(fake_image)
        return fake_image, real_image    

def get_dataloader(fake_data_dir, real_data_dir, batch_size=4, distributed=False):
    dataset = CustomDataset(fake_data_dir, real_data_dir)
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True, 
                                             )
    return dataloader
if __name__ == "__main__":
    # print(os.listdir("./dataset 2/input"))
    # for filename in os.listdir("./dataset 2/input"):
    #     print(filename)
    mydataset = CustomDataset("./dataset 2/input", "./dataset 2/ground_truth")
    data_loader = torch.utils.data.DataLoader(mydataset, batch_size=4, shuffle=True)
    for x, y in data_loader:
        print(x.shape)
        print(y.shape)

#     fake = mydataset[0][0].permute(1, 2, 0).numpy()
#     real = mydataset[0][1].permute(1, 2, 0).numpy()

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Real Image")
#     plt.imshow(real)
#     plt.axis('off')

# # Display fake image
#     plt.subplot(1, 2, 2)
#     plt.title("Fake Image")
#     plt.imshow(fake)
#     plt.axis('off')

#     plt.show()



   
