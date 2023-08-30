import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy.linalg import sqrtm
import numpy as np
from pytorch_fid import fid_score
from PIL import Image

def calculate_fid(real_images, generated_images): 
    """
    calculate fid score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def calculate_activations(images, model):
        images = images.to(device)
        print(type(images), images.shape)
        activations = model(images)
        print(type(activations), activations.shape)
        activations = adaptive_avg_pool2d(activations, output_size=(1, 1)).squeeze(3).squeeze(2)
        return activations.detach().cpu().numpy()
    
    def calculate_fid_score(real_activations, generated_activations):
        mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
        mu_generated, sigma_generated = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)
        
        diff = mu_real - mu_generated
        cov_mean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        fid = np.real(diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(cov_mean))
        return fid
    
    transform = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.ToTensor(),
        ]
    )
    real_images = Image.open(real_images).convert("RGB")
    generated_images = Image.open(generated_images).convert("RGB")
    real_images = transform(real_images) * 2 - 1
    generated_images = transform(generated_images) * 2 - 1
    # real_images = torch.nn.functional.interpolate(real_images.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
    # generated_images = torch.nn.functional.interpolate(generated_images.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)

    real_activations = calculate_activations(real_images.unsqueeze(0), inception_model)
    generated_activations = calculate_activations(generated_images.unsqueeze(0), inception_model)
    fid_score = calculate_fid_score(real_activations, generated_activations)

    return fid_score


def random_mask(images, mask_size=64):
    """
    Randomly masks images with a square mask of size `mask_size`.
    """
    masked_images = images.clone()
    batch_size, channels, height, width = images.shape
    for i in range(batch_size):
        x = torch.randint(0, width - mask_size, (1,))
        y = torch.randint(0, height - mask_size, (1,))
        masked_images[i, :, y : y + mask_size, x : x + mask_size] = 0
    return masked_images

def random_mask_single(image, mask_size=64):
    """
    Randomly masks a single image with a square mask of size `mask_size`.
    """
    masked_image = image.clone()
    channels, height, width = image.shape
    x = torch.randint(0, width - mask_size, (1,))
    y = torch.randint(0, height - mask_size, (1,))
    masked_image[:, y : y + mask_size, x : x + mask_size] = 0
    return masked_image


if __name__ == "__main__":
    fid1 = fid_score.calculate_fid_given_paths(["./realimages", "./generated"], 
                                               batch_size=1,
                                               device="cpu", 
                                               dims=2048)
    # fid2 = calculate_fid("./astronaut_rides_horse.png", "./output_image.png")
    print(f"fid1, fid2 {fid1}")