from PIL import Image
import torch
from torch._C import dtype
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

latent_size = 128
generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

generator.load_state_dict(torch.load('Generator_weights_anime_stage1.pth', map_location=torch.device('cpu')))

stats1 = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
sample_dir = 'New_Generated_Images'
os.mkdir(sample_dir)

def denormalize_the_images(image, mean, std):
  #image = open_image(image)
  if (len(image.shape) == 3):
    image = image.unsqueeze(0)
  mean = torch.tensor(mean).reshape(1,3,1,1) # image ta matrix tensor ma xa. so matrix multiply ra add garda shape eutai huna paryo nita
  std = torch.tensor(std).reshape(1,3,1,1)
  return image * std + mean

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    # denormalize_the_images(images[:nmax],*stats)
    save_image(denormalize_the_images(fake_images, stats1[0], stats1[1]), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))



for i in range(0,10):
    fixed_latent = torch.randn(64, latent_size, 1, 1)
    save_samples(i, fixed_latent, show = False)