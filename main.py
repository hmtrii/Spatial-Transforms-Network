from numpy.testing._private.utils import requires_memory
import torch
from model import STN
from dataset_utils import prepare_dataloader
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from model import STN
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import nn
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from icecream import ic
from utils import convert_image_np
import torchvision
import matplotlib.pyplot as plt

def train_one_epoch(epoch, model, train_loader, device, loss_fn, scaler, scheduler):
  model.train()
  t = time.time()
  running_loss = None
  pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training {epoch}")

  for step, (imgs, targets) in pbar:
    imgs = imgs.to(device).float()
    targets = targets.to(device).float()
    with autocast():
      pred_tensor = model(imgs) # batch_size, 3, w, h
      pred_grid = torchvision.utils.make_grid(pred_tensor).to(device)
      pred_grid.requires_grad = True
      target_grid = torchvision.utils.make_grid(targets).to(device)
      target_grid.requires_grad = True
      loss = loss_fn(pred_grid, target_grid)
      
      scaler.scale(loss).backward()

      if running_loss is None:
        running_loss = loss.item()
      else:
        running_loss = running_loss * 0.99 + loss.item() * 0.01
      
      if step + 1 == len(train_loader):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() 
        scheduler.step()

        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)
  
  return running_loss

def val_one_epoch():
  pass

if __name__ == '__main__':
  # parameters
  img_prefix = r'd:\UIT\ChestXray\data\train'
  batch_size = 16
  num_workers = 2
  img_size = 512
  lr = 1e-5
  weight_decay = 1e-6
  T_0 = 10
  min_lr = 1e-6
  num_epochs = 50
  canonical_path = './canonical.png'

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = STN().to(device)
  df = pd.read_csv(r'd:\UIT\ChestXray\data\stratified5folds.csv')
  train_df, val_df = train_test_split(df, test_size=0.2)
  train_loader, val_loader = prepare_dataloader(train_df, val_df, img_prefix, img_size, batch_size, num_workers, canonical_path=canonical_path)
  scaler = GradScaler()
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
  num_epoche = num_epochs
  loss_fn = nn.MSELoss().to(device)

  for epoch in range(1, num_epochs + 1):
    train_one_epoch(epoch, model, train_loader, device, loss_fn, scaler, scheduler)
    aa
    with torch.no_grad():
      val_one_epoch()