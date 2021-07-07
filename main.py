import torch
from model import STN
from dataset_utils import prepare_dataloader
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

def train_one_epoch():
  pass

def val_one_epoch():
  pass

if __name__ == '__main__':
  # parameters
  img_prefix = r'd:\UIT\ChestXray\data\train'
  batch_size = 16
  num_workers = 2
  img_size = 512

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # model = STN().to(device)
  df = pd.read_csv(r'd:\UIT\ChestXray\data\stratified5folds.csv')
  train_df, val_df = train_test_split(df, test_size=0.2)
  train_loader, val_loader = prepare_dataloader(train_df, val_df, img_prefix, img_size, batch_size, num_workers)
  
  for img in train_loader:
    print(img.shape)
    break