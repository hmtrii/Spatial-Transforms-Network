import os
import random
import numpy as np
import torch
import cv2
import copy
from albumentations import (
    Resize, Compose, Normalize, Resize
)
from albumentations.pytorch import ToTensorV2
from torch.utils import data
import torchvision
from tqdm import tqdm

def seed_torch(seed=2021):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def convert_image_np(inp):
  """Convert a Tensor to numpy image."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  return inp

def get_img(path):
  im_bgr = cv2.imread(path)
  im_rgb = im_bgr[:, :, ::-1]
  return im_rgb

def get_transforms(img_size):
  return Compose([
          Resize(img_size, img_size, always_apply=True),
          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
          ToTensorV2(p=1.0),
          ], p=1.)

def train_model(model, num_epochs, dataloader, device, criterion, optimizer, scheduler, scaler=None):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 0
  for epoch in range(1, num_epochs + 1):
    last_model, best_model = run_one_epoch(epoch, model, dataloader, device, criterion, optimizer, scheduler, best_loss, best_model_wts, scaler)
  
  return last_model, best_model

def run_one_epoch(epoch, model, dataloader, device, criterion, optimizer, scheduler, best_loss, best_model_wts, scaler=None):
  best_model = copy.deepcopy(model)
  for phase in ['train', 'val']:
    if phase == 'train':
      model.train()
    else:
      model.eval()
    
    running_loss = 0
    pbar = tqdm(dataloader[phase], total=len(dataloader), desc=f"{phase}: {epoch}")
    for imgs, targets in pbar:
      imgs = imgs.to(device).float()
      targets = targets.to(device).float()

      optimizer.zero_grad()
      
      with torch.set_grad_enabled(phase == 'train'):
        pred_tensor = model(imgs)
        pred_grid = torchvision.utils.make_grid(pred_tensor).to(device).float()
        pred_grid.requires_grad = True
        target_grid = torchvision.utils.make_grid(targets).to(device).float()
        target_grid.requires_grad = True

        loss = criterion(pred_grid, target_grid)

        if phase == 'train':
          if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
          else:
            loss.backward()
            optimizer.step()

      running_loss += loss.item() * imgs.size(0)

    if phase == 'train':
      scheduler.step()
    
    epoch_loss = running_loss / len(dataloader[phase])
    description = f'epoch {epoch} loss: {epoch_loss:.4f}'
    pbar.set_description(description)

    if phase == 'val' and epoch_loss < best_loss:
      best_loss = epoch_loss
      best_model_wts = copy.deepcopy(model.state_dict())
      best_model.load_state_dict(best_model_wts)

  return model, best_model
