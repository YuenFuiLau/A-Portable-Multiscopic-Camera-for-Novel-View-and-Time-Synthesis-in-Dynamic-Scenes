import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#from kornia.filters import spatial_gradient

from torch.autograd import Variable
from math import exp

#SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True,device=torch.device("cuda:0")):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)#.cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)#.cuda()
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.resize = resize

    def forward(self, syn_imgs, gt_imgs):
        syn_imgs = (syn_imgs - self.mean) / self.std
        gt_imgs = (gt_imgs - self.mean) / self.std
        if self.resize:
            syn_imgs = self.transform(syn_imgs, mode="bilinear", size=(224, 224),
                                      align_corners=False)
            gt_imgs = self.transform(gt_imgs, mode="bilinear", size=(224, 224),
                                     align_corners=False)

        loss = 0.0
        x = syn_imgs
        y = gt_imgs
        for block in self.blocks:
            with torch.no_grad():
                x = block(x)
                y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def psnr(img1, img2):
    #mse = ((img1 - img2) ** 2).mean((1, 2, 3))
    mse = ((img1 - img2) ** 2).mean()
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    #psnr.mean()
    return psnr


"""
def edge_aware_loss(img, disp, gmin, grad_ratio):
    # Compute img grad and grad_max
    grad_img = torch.abs(spatial_gradient(img)).sum(1, keepdim=True).to(torch.float32)
    grad_img_x = grad_img[:, :, 0]
    grad_max_x = torch.amax(grad_img_x, dim=(1, 2, 3), keepdim=True)
    grad_img_y = grad_img[:, :, 1]
    grad_max_y = torch.amax(grad_img_y, dim=(1, 2, 3), keepdim=True)

    # Compute edge mask
    edge_mask_x = grad_img_x / (grad_max_x * grad_ratio)
    edge_mask_y = grad_img_y / (grad_max_y * grad_ratio)
    edge_mask_x = torch.where(edge_mask_x < 1, edge_mask_x, torch.ones_like(edge_mask_x).cuda())
    edge_mask_y = torch.where(edge_mask_y < 1, edge_mask_y, torch.ones_like(edge_mask_y).cuda())

    # Compute and normalize disp grad
    grad_disp = torch.abs(spatial_gradient(disp, normalized=False))
    grad_disp_x = F.instance_norm(grad_disp[:, :, 0])
    grad_disp_y = F.instance_norm(grad_disp[:, :, 1])

    # Compute loss
    grad_disp_x = grad_disp_x - gmin
    grad_disp_y = grad_disp_y - gmin
    loss_map_x = torch.where(grad_disp_x > 0.0, grad_disp_x,
                             torch.zeros_like(grad_disp_x).cuda()) * (1.0 - edge_mask_x)
    loss_map_y = torch.where(grad_disp_y > 0.0, grad_disp_y,
                             torch.zeros_like(grad_disp_y).cuda()) * (1.0 - edge_mask_y)
    return (loss_map_x + loss_map_y).mean()
"""

def edge_aware_loss_v2(img, disp):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(2, True).mean(3, True)
    disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def edge_aware_loss(img, disp):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness

    param disp : (H,W) 
    param img : (H,W,C)
    """
    mean_disp = disp.mean(0, True).mean(1, True)
    normalize_disp = disp / (mean_disp + 1e-7)
    #disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(normalize_disp[:, :-1] - normalize_disp[ :, 1:]) #(H,W-1)
    grad_disp_y = torch.abs(normalize_disp[:-1, :] - normalize_disp[ 1:, :]) #(H-1,W)

    grad_img_x = torch.mean(torch.abs(img[:, :-1,:] - img[:, 1:,:]), dim=-1, keepdim=False) #(H,W-1)
    grad_img_y = torch.mean(torch.abs(img[:-1, :,:] - img[1:, :,:]), dim=-1, keepdim=False) #(H-1,W)

    grad_disp_x *= torch.exp(-grad_img_x) #(H,W-1)
    grad_disp_y *= torch.exp(-grad_img_y) #(H-1,W)

    return grad_disp_x.mean() + grad_disp_y.mean()


def edge_sharpen_loss(img, dep):

    """
    param dep : (H,W) 
    param img : (H,W,C)
    """
    #depth
    depth_normalizing_constant = torch.max(dep).clone().detach()
    normalize_disp = dep / (depth_normalizing_constant + 1e-7)

    grad_dep_x = torch.abs(normalize_disp[:,1:] - normalize_disp[:,:-1]) #(H,W-1)
    grad_dep_y = torch.abs(normalize_disp[1:,:] - normalize_disp[:-1,:]) #(H-1,W)

    #img
    img_normalizing_constant = torch.max(img).clone().detach()
    normalize_img = img / (img_normalizing_constant + 1e-7)

    grad_img_x = torch.mean(torch.abs(normalize_img[:,1:,:] - normalize_img[:,:-1,:]), dim=-1, keepdim=False )#(H,W-1)
    grad_img_y = torch.mean(torch.abs(normalize_img[1:,:,:] - normalize_img[:-1,:,:]), dim=-1, keepdim=False )#(H-1,W)

    grad_result = torch.abs(grad_dep_x - grad_img_x).mean() + torch.abs(grad_dep_y - grad_img_y).mean()

    return grad_result
