import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)
        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        #inputs = F.normalize(inputs, mean, std)
        inputs = transforms.Normalize(mean,std)(inputs)
        #targets = F.normalize(targets, mean, std)
        targets = transforms.Normalize(mean,std)(targets)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        
        # compare their weighted loss
        b = len(input_features)
        for i in range(b):

            for lhs, rhs, w in zip(input_features[i], target_features[i], self.weights):
                #lhs = lhs.view(lhs.size(0), -1)
                #rhs = rhs.view(rhs.size(0), -1)
                loss += self.feature_loss(lhs, rhs) * w

        return loss

def perceptual_loss(x, y):
    F.mse_loss(x, y)
    
def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

def gram_matrix(x):
    c, h, w = x.size()
    x = x.view(c, -1)
    x = torch.mm(x, x.t()) / (c * h * w)
    return x

def gram_loss(x, y):
    return F.mse_loss(gram_matrix(x), gram_matrix(y))

def TextureLoss(blocks, weights, device):
    return FeatureLoss(gram_loss, blocks, weights, device)

if __name__ == "__main__":

    loss = TextureLoss(blocks=[1,2,3,4], weights=[1.0,1.0,1.0,1.0], device="cuda")
    a = torch.randn(1,3,224,224)
    b = torch.randn(1,3,224,224)

    k = loss(a.cuda(),b.cuda())

    print(k)
    
