import os
import csv
import shutil
from glob import glob
from tqdm import tqdm

import argparse

print("Process initiated...")

def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', '-m', type=str, default="audio", choices=['audio', 'roi', 'mp4'])
    parser.add_argument('--move', action='store_true', help='Move files to new location')
    args = parser.parse_args()
    return args

args = load_args()
modality = args.modality
all_files = glob(f'./datasets/ndc-me/{modality}/data/*.npz', recursive=True)

dev_data, test_data = [], []

for file in tqdm(all_files):
    split = file.split(os.path.sep)
    print(split)
    path = os.path.join(*split[:4], 'data', split[-1])
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    if split[-2] == 'test':
        test_data.append(path)
    else:
        dev_data.append(path)

    if args.move:
        shutil.move(file, path)

with open(f'./datasets/ndc-me/{modality}/dev_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for file in dev_data:
        writer.writerow([file])

with open(f'./datasets/ndc-me/{modality}/test_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for file in test_data:
        writer.writerow([file])

print("Done!")