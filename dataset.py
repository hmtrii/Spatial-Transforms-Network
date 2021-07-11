import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_img
import os

class CXRDataset(Dataset):
  def __init__(self, df, img_prefix, canonical_path=None, transforms=None):
    super(CXRDataset, self).__init__()
    self.df = df
    self.img_prefix = img_prefix
    self.transform = transforms
    self.canonical_path = canonical_path

  def __len__(self):
    return self.df.shape[0]
  
  def __getitem__(self, index):
    image = get_img(os.path.join(self.img_prefix, self.df.iloc[index]['image_id']))
    image = self.transform(image=image)['image']
    if self.canonical_path:
      target = get_img(self.canonical_path)
      target = self.transform(image=target)['image']
      return image, target
    return image
