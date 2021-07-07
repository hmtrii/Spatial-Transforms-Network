from random import shuffle
from dataset import CXRDataset
from utils import get_transforms
from torch.utils.data import DataLoader

def prepare_dataloader(train_df, val_df, img_prefix, img_size, batch_size, num_wokers):
  train_dataset = CXRDataset(train_df, img_prefix, transforms=get_transforms(img_size))
  val_dataset = CXRDataset(val_df, img_prefix, transforms=get_transforms(img_size))

  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_wokers
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_wokers
  )
  
  return train_loader, val_loader