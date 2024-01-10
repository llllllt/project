# build a U-net model
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class DouboleConv(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.b1 = nn.BatchNorm2d(output_dim)
        self.b2 = nn.BatchNorm2d(output_dim)
    def forward(self, x):
        x = F.silu(self.b1(self.conv1(x)))
        h = F.silu(self.b2(self.conv2(x)))
        return x + h

class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()  
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.pool(x)
        return x

class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_dim, output_dim, 2, stride=2)
        self.b1 = nn.BatchNorm2d(output_dim)
    def forward(self, x):
        x = self.up(x)
        x = F.silu(self.b1(x))
        return x

class CropAndConcat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, contracting_x):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        return torch.cat([x, contracting_x], dim=1)


 
class Unet(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_list = [in_dim, 64, 128, 256, 512, 1024]

        self.down_conv = nn.ModuleList([DouboleConv(i, o) for i, o in zip(self.in_dim_list[:-2], self.in_dim_list[1:-1])])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(len(self.in_dim_list)-2)])
        self.middle_conv = DouboleConv(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in zip(self.in_dim_list[::-1][:-2], self.in_dim_list[::-1][1:-1])])
        self.up_conv = nn.ModuleList([DouboleConv(i, o) for i, o in zip(self.in_dim_list[::-1][:-2], self.in_dim_list[::-1][1:-1])])
        self.crop_and_concat = nn.ModuleList([CropAndConcat() for _ in range(len(self.in_dim_list)-2)])
        self.out_conv = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        hidden_list = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            hidden_list.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_sample)):
            x = self.up_sample[i](x)
            x = self.crop_and_concat[i](x, hidden_list.pop())
            x = self.up_conv[i](x)
        x = self.out_conv(x)
        return x
    

if __name__ == "__main__":
    model = Unet(3, 3)
    print(model)
