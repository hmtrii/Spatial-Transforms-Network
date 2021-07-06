import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_img
import os

class CXRDataset(Dataset):
  def __init__(self, df, img_prefix, transform):
    super(CXRDataset, self).__init__()
    self.df = df
    self.img_prefix = img_prefix
    self.transform = transform
  
  def __len__(self):
    return self.df.shape[0]
  
  def __getitem__(self, index):
    image = get_img(os.path.join(self.img_prefix + self.df.iloc[index]['image_id']))
    image = self.transform(image)
    return image
