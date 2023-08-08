import torch

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