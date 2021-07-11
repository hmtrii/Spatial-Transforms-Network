from torch.utils import data
from dataset import CXRDataset
from utils import get_transforms
from torch.utils.data import DataLoader

def prepare_dataloader(train_df, val_df, img_prefix, img_size, batch_size, num_wokers,canonical_path=None):
  train_dataset = CXRDataset(train_df, img_prefix, canonical_path=canonical_path, transforms=get_transforms(img_size))
  val_dataset = CXRDataset(val_df, img_prefix, canonical_path=canonical_path, transforms=get_transforms(img_size))

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
  
  dataloaders = {'train': train_loader,
                  'val': val_loader}

  return dataloaders