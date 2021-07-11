import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class STN(nn.Module):
  def __init__(self):
    super(STN, self).__init__()
    self.backbone = models.resnet18()
    self.backbone.fc = nn.Linear(512, 3 * 2)
    self.backbone.fc.weight.data.zero_()
    self.backbone.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
  
  def localization(self, x):
    return self.backbone(x)

  def stn(self, x):
    theta_vec = self.localization(x).view(-1, 2, 3)
    grid = F.affine_grid(theta_vec, x.size(), align_corners=True)
    return F.grid_sample(x, grid, align_corners=True)

  def forward(self, x):
    return self.stn(x)
