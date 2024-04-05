from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import datetime
from graph.block_diag_matrix import block_diag
import random

from utils import get_data_dir

"""
    Bands: [1: AOT, 2: B02, 3: B03, 4:B04, 5: B05, 6: B06, 7: B07, 8: B8A, 9: WVP(B09),
            10: B11, 11: B12, 12: CLD, 13: SNW, 14: SCL, 15: LST_DAY, 16: LST_NIGHT, 17: RAIN]
"""


def haversine_fn(latlong):
    """
    Calculate the haversine distance starting from lat and long
    :param latlong: lat and long
    :return:
    """
    R = 6373
    data = np.deg2rad(latlong)
    lat = data[:, 0]
    lng = data[:, 1]
    diff_lat = lat[:, None] - lat
    diff_lng = lng[:, None] - lng

    a = np.sin(diff_lat / 2) ** 2 + \
        np.cos(lat[:, None]) * np.cos(lat) * np.sin(diff_lng / 2)**2
    distances2 = 2 * R * np.arcsin(np.sqrt(a))
    return distances2


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


def compute_adj_matrix_aux(aux_bands):
    """
    Compute adjacency matrix from a tensor of n images.
    """
    s = 1.
    l1 = nn.L1Loss()
    # compute the distances between all aux data in the current batch
    distances = [[l1(a, b) for a in aux_bands] for b in aux_bands]
    distances = list(map(torch.stack, distances))
    distances = torch.stack(distances)
    # compute gaussian kernel in order to obtain a similarity matrix
    adj_matrix = torch.exp(- (distances) / (2. * s ** 2))
    # normalization choice
    return adj_matrix


def id_collate(batch):
    imgs_neighbours = torch.cat([list(_batch)[0] for _batch in batch], dim=0)
    labels = torch.tensor([_batch[1] for _batch in batch]).view(-1)
    if len(batch[0][2].size()) == 3:
        adj_matrix = torch.stack([_batch[2] for _batch in batch], dim=1)
        blocks_adj_matrix = [block_diag(adj)for adj in adj_matrix]
        blocks_adj_matrix = torch.stack(blocks_adj_matrix, 0)
    else:
        adj_matrix = torch.stack([_batch[2] for _batch in batch], dim=0)
        blocks_adj_matrix = block_diag(adj_matrix)
    date_infection = torch.cat([_batch[-3] for _batch in batch], dim=0)
    company_cod = [_batch[-2] for _batch in batch]
    img_dates = [_batch[-1] for _batch in batch]
    return imgs_neighbours, labels, blocks_adj_matrix, date_infection, company_cod, img_dates


class AM_Dataset(Dataset):
    def __init__(self, args, path_label, bands=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dsize=224,
                 temporal_buffer=0, train=False,
                 random_seed=42, random_temporal=0, eval_last_year=False):
        self.args = args
        self.train = train
        self.eval_last_year = eval_last_year
        self.temporal_buffer = temporal_buffer
        self.label_threshold = args.label_threshold if hasattr(args, 'label_threshold') else 0
        # saving the current dir of this subfolder for local imports
        self.curr_dir = Path(__file__).parent
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # unzip path and label
        self.path_imgs, self.labels, self.dates, self.companies_cod, self.paths_images_neigh, \
            self.labels_neigh, self.latlong, self.latlong_neigh = zip(*path_label)
        self.labels = (np.stack(self.labels, axis=0) > self.label_threshold).astype(int)

        # ----- Fix datetime order
        dates_dt = np.array([datetime.datetime.strptime(i.split('/')[-1], '%Y%m%d') for i in self.dates])
        year_dt = np.array([datetime.datetime.strptime(i.split('/')[-1], '%Y%m%d').year for i in self.dates])

        if eval_last_year:
            last_year = np.max(year_dt)
            mask = year_dt < last_year if train else year_dt == last_year

            self.path_imgs = np.array(self.path_imgs)[mask]
            self.labels = self.labels[mask]
            self.dates = np.array(self.dates)[mask]
            self.companies_cod = np.array(self.companies_cod)[mask]
            self.paths_images_neigh = np.array(self.paths_images_neigh)[mask]
            self.labels_neigh = np.array(self.labels_neigh)[mask]
            self.latlong = np.array(self.latlong)[mask]
            self.latlong_neigh = np.array(self.latlong_neigh)[mask]
            dates_dt = dates_dt[mask]

        order = np.argsort(dates_dt)
        self.path_imgs = np.array(self.path_imgs)[order]
        self.labels = self.labels[order]
        self.dates = np.array(self.dates)[order]
        self.companies_cod = np.array(self.companies_cod)[order]
        self.paths_images_neigh = np.array(self.paths_images_neigh)[order]
        self.labels_neigh = np.array(self.labels_neigh)[order]
        self.latlong = np.array(self.latlong)[order]
        self.latlong_neigh = np.array(self.latlong_neigh)[order]

        data_len = len(self.path_imgs)

        # ----- Preprocessing as TS

        # Group by latlong (used as ID)
        unique_latlongs, unique_latlong_counts = np.unique(self.latlong, axis=0, return_counts=True)

        split_data_map = {}
        valid_sampling_locations = []
        all_idxs = np.arange(data_len)

        for latlong, cnt in zip(unique_latlongs, unique_latlong_counts):
            identifier = tuple(latlong)
            mask = (latlong == self.latlong).all(1)
            valid_idxs = all_idxs[mask]
            num_entries = len(valid_idxs)

            assert num_entries == cnt, f"{num_entries} != {cnt} for {identifier}"
            if num_entries >= temporal_buffer:
                valid_sampling_locations += [
                    (identifier, i)
                    for i in range(num_entries - temporal_buffer)
                ]
                split_data_map[identifier] = valid_idxs

        self.split_data_map = split_data_map
        self.valid_sampling_locations = valid_sampling_locations
        self.data_len = len(valid_sampling_locations)

        # ----- Updating paths
        self.data_path = get_data_dir()

        # ----- Finalize
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
        self.days_diff_total = []
        if random_seed is not None:
            random.seed(random_seed)
        self.random_temporal = random_temporal

        self.bins = (0, 100, 200, 300, 400, 500, 650, 800, 1000, 1500, 2000)

    def load_tensor_image(self, imgs_file: str) -> torch.Tensor:
        spectral_img = torch.load(imgs_file + '/bands.pt')
        spectral_img = torch.squeeze(
            nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.dsize))

        if spectral_img.shape[0] == 17:
            spectral_img = spectral_img[:-1]

        # choose the bands to keep
        spectral_img = spectral_img[self.bands]
        return spectral_img

    def __getitem__(self, index):
        identifier, start_idx = self.valid_sampling_locations[index]
        index, label_idx = self.split_data_map[identifier][start_idx], self.split_data_map[identifier][start_idx + self.temporal_buffer]

        # this version exploit all the available images, so no temporal buffer or other...
        # 1- open the current image and the ones of the neighborhood
        # (-1 for take the last image, eventually apply temporal buffer)
        imgs_file = self.path_imgs[index]
        spectral_img = self.load_tensor_image(imgs_file=imgs_file)

        # 2- open the neighbours images
        imgs_neighbours = [spectral_img]
        label = self.labels[label_idx][0]
        latlong_neigh = [self.latlong[index]]
        for idx, n in enumerate(self.paths_images_neigh[index]):
            imgs_neighbours.append(self.load_tensor_image(n))
            latlong_neigh.append(self.latlong_neigh[index][idx])
        imgs_neighbours = torch.stack(imgs_neighbours, dim=0)
        latlong_neigh = np.stack(latlong_neigh, axis=0)

        adj_matrix = compute_adj_matrix_aux(imgs_neighbours[:, -1].mean(dim=[1, 2])).unsqueeze(dim=0)
        imgs_neighbours = imgs_neighbours[:, :-1]

        # fix satellite underflow
        if imgs_neighbours[:, -1].min() < -1e30:
            imgs_neighbours[:, -1] = torch.maximum(torch.zeros_like(imgs_neighbours[:, -1]), imgs_neighbours[:, -1])

        return imgs_neighbours, torch.tensor(label), adj_matrix, \
            datemonth_to_month(self.dates[index], N=len(imgs_neighbours)), self.companies_cod[index], self.dates[index]

    def __len__(self):
        return self.data_len
