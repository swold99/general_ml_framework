import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch

folder = os.path.join('data', 'images')
label_dict = {}

for filename in tqdm(os.listdir(folder)):
    if 'xml' in filename:
        with open(os.path.join(folder, filename), "r") as f:
            data = f.read()
            Bs_data = BeautifulSoup(data, 'xml')
            coords = []
            for object in Bs_data.find_all('bndbox'):
                coords.append([int(coord) for coord in object.text.strip().split('\n')]) #xmin, ymin, xmax, ymax
            label_dict[filename.split('.')[0]] = coords

for name, coords in label_dict.items():
    a = True
    if len(coords) > 1:
        print(len(coords), name)
        a=False

print("anus")
