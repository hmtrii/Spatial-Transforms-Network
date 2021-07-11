import os
import torch
import argparse
import pandas as pd
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import STN
from utils import train_model
from dataset_utils import prepare_dataloader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
  # parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_prefix', type=str, default='./dataset/train/', help='Image directory')
  parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
  parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
  parser.add_argument('--img-szie', type=int, default=512, help='Image size')
  parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
  parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay')
  parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimial learning rate')
  parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
  parser.add_argument('--canonical-path', type=str, default='./dataset/canonical/canonical.png')
  parser.add_argument('--work-dir', type=str, default='./runs', help='Directory of result')
  opt = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = STN().to(device)
  df = pd.read_csv(r'd:\UIT\ChestXray\data\stratified5folds.csv')
  train_df, val_df = train_test_split(df, test_size=0.2)
  train_loader, val_loader = prepare_dataloader(train_df, val_df, opt.img_prefix, opt.img_size, opt.batch_size, opt.num_workers, canonical_path=opt.canonical_path)
  scaler = GradScaler()
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=opt.min_lr, last_epoch=-1)
  num_epochs = opt.num_epochs
  criterion = nn.MSELoss().to(device)

  dataloaders = prepare_dataloader(train_df, val_df, opt.img_prefix, opt.img_size, opt.batch_size, opt.num_workers, opt.canonical_path)

  last_model, best_model = train_model(model, num_epochs, dataloaders, device, criterion, optimizer, scheduler)
  torch.save(last_model.state_dict(), os.join(opt.work_dir, 'last_model.pth'))
  torch.save(best_model.state_dict(), os.join(opt.work_dir, 'best_mode.pth'))
