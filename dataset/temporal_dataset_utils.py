import json
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np


class DatasetUtils:
    def __init__(self, json_file, use_custom_datapath=False, num_multi_images=10, random_seed=42, n_splits=5):
        # check if the json file exist
        self.num_multi_images = num_multi_images
        self.use_custom_datapath = use_custom_datapath
        assert os.path.isfile(json_file), "No json configuration file found at {}".format(json_file)
        # open the json file
        with open(json_file) as f:
            self.json_db = json.load(f)
            self.json_db = self.json_db['companies']
        self.random_seed = random_seed

        self.labels_companies_approx = [1 if np.mean([sample['labels'][0] > 0 for sample in aziend['samples']]) > 0.5 else 0 for aziend in self.json_db]

        self.train_indices, self.test_indices = [], []
        # the shuffle split can have repeated or missing values!

        # remove samples with less than 10 images
        self.json_db = [{'company_cod': company['company_cod'],
                         'samples': [sample for sample in company['samples'] if len(sample['imgs']) >= self.num_multi_images],
                         'latitude': company['latitude'],
                         'longitude': company['longitude']} for company in self.json_db]

        if n_splits == 1:
            self.train_indices = [list(range(len(self.json_db)))]
            self.test_indices = [list(range(len(self.json_db)))]
        else:
            strat = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

            for train_idx, test_idx in strat.split(self.json_db, self.labels_companies_approx):
                self.train_indices.append(train_idx)
                self.test_indices.append(test_idx)
        print("END")

    def paths_and_labels(self, split=0):
        self.paths_labels_train = self.split_abr_mol(mode='train', split=split)
        self.paths_labels_test = self.split_abr_mol(mode='test', split=split)
        return self.paths_labels_train, self.paths_labels_test

    def split_abr_mol(self, mode='train', split=0):
        # iterate over the companies in the train set
        paths_images = []
        labels = []
        dates = []
        companies_cod = []
        if mode == 'train':
            indices = self.train_indices[split]
        else:
            indices = self.test_indices[split]
        for i in indices:
            for sample in self.json_db[i]['samples']:
                if len(sample['imgs']) < self.num_multi_images:
                    continue
                paths_images.append(sample['imgs'])
                labels.append(sample['labels'])
                dates.append(sample['date'])
                companies_cod.append(self.json_db[i]['company_cod'])
        return list(zip(paths_images, labels, dates, companies_cod))
