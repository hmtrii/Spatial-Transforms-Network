import cv2 
import glob
import pandas as pd
from icecream import ic
import numpy as np
from tqdm import tqdm

all_images = []
df = pd.read_csv(r'd:\UIT\ChestXray\data\stratified5folds.csv')
df = df.loc[df["image_id"]!=13].sample(n=500, replace=False)
for i, row in tqdm(df.iterrows(), total=500):
  if row.class_id != 13:
    all_images.append(cv2.imread(rf'd:\UIT\ChestXray\data\train\{row.image_id}'))

# img1 = cv2.imread(r'd:\UIT\ChestXray\data\train\36201.png')
# img2 = cv2.imread(r"d:\UIT\ChestXray\data\train\36202.png")
# ic(type(img1))
# ic(img2)
img = np.mean(all_images, axis=0)
cv2.imshow('before', img.astype(np.uint8))
kernel = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])

img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
cv2.imshow('after', img.astype(np.uint8))
cv2.waitKey(0)