import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoencoder(nn.Module):
    """No pretrained weights, external features, or bottleneck bypass."""
    def __init__(self, latent_dim=128):
        super().__init__()
        channels = (1,16,32,64,96,128)
        blocks = []
        for cin,cout in zip(channels[:-1],channels[1:]):
            blocks.extend([nn.Conv2d(cin,cout,3,stride=2,padding=1), nn.GroupNorm(8,cout), nn.SiLU()])
        self.encoder = nn.Sequential(*blocks)
        self.to_latent = nn.Linear(128*8*8, latent_dim)
        self.from_latent = nn.Linear(latent_dim,128*8*8)
        self.decoder = nn.ModuleList()
        for cin,cout in zip((128,96,64,32,16),(96,64,32,16,1)):
            layers = [nn.Conv2d(cin,cout,3,padding=1)]
            layers += [nn.Sigmoid()] if cout == 1 else [nn.GroupNorm(8,cout),nn.SiLU()]
            self.decoder.append(nn.Sequential(*layers))

    def encode(self,x):
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self,z):
        x = self.from_latent(z).reshape(-1,128,8,8)
        for size,block in zip((15,29,57,114,227),self.decoder):
            x = block(F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False))
        return x

    def forward(self,x):
        z = self.encode(x)
        return self.decode(z),z


def ssim_per_image(x,y):
    """11x11 Gaussian-window SSIM, sigma 1.5, data range 1, valid windows."""
    x,y = x.float(),y.float()
    coords = torch.arange(11,device=x.device,dtype=x.dtype)-5
    weights = torch.exp(-coords.square()/(2*1.5**2))
    weights = weights / weights.sum()
    # Separable filters compute the same Gaussian window with less work.
    def smooth(a):
        a = F.conv2d(a,weights.reshape(1,1,1,11))
        return F.conv2d(a,weights.reshape(1,1,11,1))
    mx,my = smooth(x),smooth(y)
    vx = (smooth(x*x)-mx*mx).clamp_min(0)
    vy = (smooth(y*y)-my*my).clamp_min(0)
    cov = smooth(x*y)-mx*my
    result = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return result.mean((1,2,3))


def loss_components(x,reconstruction,structural_weight=0.,gradient_weight=0.):
    x,reconstruction = x.float(),reconstruction.float()
    mse = (x-reconstruction).square().mean((1,2,3))
    with torch.set_grad_enabled(torch.is_grad_enabled() and structural_weight>0):
        ssim = ssim_per_image(x,reconstruction)
    dx = (x[:,:,:,1:]-x[:,:,:,:-1])-(reconstruction[:,:,:,1:]-reconstruction[:,:,:,:-1])
    dy = (x[:,:,1:,:]-x[:,:,:-1,:])-(reconstruction[:,:,1:,:]-reconstruction[:,:,:-1,:])
    gradient = .5*(dx.abs().mean((1,2,3))+dy.abs().mean((1,2,3)))
    loss = (1-structural_weight)*mse
    if structural_weight: loss = loss+structural_weight*(1-ssim)
    if gradient_weight: loss = loss+gradient_weight*gradient
    return {'loss':loss.mean(),'mse':mse.mean(),'ssim':ssim.mean(),'gradient':gradient.mean()}
