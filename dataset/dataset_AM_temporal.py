from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import datetime

from tqdm import tqdm
from utils import get_data_dir

"""
    Bands: [1: AOT, 2: B02, 3: B03, 4:B04, 5: B05, 6: B06, 7: B07, 8: B8A, 9: WVP(B09),
            10: B11, 11: B12, 12: CLD, 13: SNW, 14: SCL]
"""


def datemonth_to_month(date_month, N=1):
    """
    Convert a date in the format YYYYMMDD to a one-hot vector for the month.
    """
    month = int(date_month[4:6]) - 1

    if N > 1:
        month = torch.ones(N) * month
    else:
        month = torch.tensor(month)

    return month.long()


def id_collate(batch):
    vals, labels = [], []
    date = []
    img_dates = []
    company_cod = []
    for _batch in batch:
        vals.append(_batch[0])
        labels.append(_batch[1])
        date.append(_batch[-3])
        company_cod.append(_batch[-2])
        img_dates.append(_batch[-1])
    return torch.stack(vals, dim=0), torch.stack(labels, dim=0), torch.stack(date, dim=0), company_cod, img_dates


class AM_Dataset(Dataset):
    def __init__(self, args, path_label, bands=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dsize=224,
                 num_multi_images=1, temporal_buffer=0, train=False, eval_last_year=False):
        assert num_multi_images > 0
        self.args = args
        self.train = train
        self.eval_last_year = eval_last_year
        self.label_threshold = args.label_threshold if hasattr(args, 'label_threshold') else 0
        # saving the current dir of this subfolder for local imports
        self.curr_dir = Path(__file__).parent
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # unzip path and label
        self.path_imgs, self.labels, self.dates, self.companies_cod = zip(*path_label)
        self.path_imgs = list(self.path_imgs)

        self.labels = (np.stack(self.labels, axis=0) > self.label_threshold).astype(int)
        year_dt = np.array([datetime.datetime.strptime(i.split('/')[-1], '%Y%m%d').year for i in self.dates])

        if eval_last_year:
            last_year = np.max(year_dt)
            mask = year_dt < last_year if train else year_dt == last_year

            self.path_imgs = np.array(self.path_imgs)[mask]
            self.labels = np.array(self.labels)[mask]
            self.dates = np.array(self.dates)[mask]
            self.companies_cod = np.array(self.companies_cod)[mask]

        self.data_path = get_data_dir()

        # Calculate len
        self.data_len = len(self.path_imgs)
        print("Dataset len: ", self.data_len)
        lab_unq, lab_counts = np.unique(self.labels, return_counts=True)
        if len(lab_unq) > 1:
            print(f"Labels. [{lab_unq[0]}]: {lab_counts[0]} | [{lab_unq[1]}]: {lab_counts[1]}")
        print("Per year")
        print(pd.to_datetime(self.dates, format='%Y%m%d').year.value_counts())
        # bands
        self.bands = torch.BoolTensor(bands)
        # image size
        self.dsize = dsize
        # choose how many temporal images
        self.num_multi_images = num_multi_images
        # single or multi-label
        self.temporal_buffer = temporal_buffer
        self.days_diff_total = []

        self.bins = (0, 100, 200, 300, 400, 500, 650, 800, 1000, 1500, 2000)

        for idx, (d, p) in tqdm(enumerate(zip(self.dates, self.path_imgs)), desc='Reoder time data', total=len(self.dates)):
            timediffs = [(datetime.datetime.strptime(v.split('/')[-1], '%Y%m%d') - datetime.datetime.strptime(d, '%Y%m%d')).days for v in p]

            reorder = np.argsort(timediffs)
            p = np.array(p)
            self.path_imgs[idx] = p[reorder]

    def __getitem__(self, index):
        # obtain the right folder
        temporal_images = []
        temporal_dates = []

        for n_img in range(self.num_multi_images, 0, -1):
            imgs_file = self.path_imgs[index][-n_img - self.temporal_buffer]
            c_date = datemonth_to_month(imgs_file.split("/")[-1])
            temporal_dates.append(c_date)

            spectral_img = torch.load(imgs_file + '/bands.pt')
            spectral_img = torch.squeeze(
                nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.dsize))
            # keep only the selected bands
            if spectral_img.shape[0] == 17:
                spectral_img = spectral_img[:-1]

            spectral_img = spectral_img[self.bands]
            if self.train:
                spectral_img = self.custom_augmentation(spectral_img)
            temporal_images.append(spectral_img)

        if self.num_multi_images != 1:
            temporal_images = torch.stack(temporal_images, dim=0)
        else:
            temporal_images = spectral_img

        return temporal_images, torch.tensor(self.labels[index][0]), \
            torch.tensor(temporal_dates), self.companies_cod[index], self.dates[index]

    def __len__(self):
        return self.data_len

    def custom_augmentation(self, images):
        rnd = np.random.random_sample()
        images = torch.unsqueeze(images, dim=1)
        angle = np.random.randint(-15, 15)
        for id, image in enumerate(images):
            if rnd < 0.25:
                image = TF.to_pil_image(image)
                image = TF.rotate(image, angle)
            elif 0.25 <= rnd <= 0.50:
                image = TF.to_pil_image(image)
                image = TF.vflip(image)
            elif 0.50 < rnd <= 0.75:
                image = TF.to_pil_image(image)
                image = TF.hflip(image)
            else:
                images = torch.squeeze(images)
                return images
            images[id] = TF.to_tensor(image)
        return torch.squeeze(images)
