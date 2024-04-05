import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


class DatasetUtils:
    def __init__(self, json_file, random_seed=42, n_splits=5):
        # check if the json file exist
        assert os.path.isfile(json_file), "No json configuration file found at {}".format(json_file)
        # open the json file
        with open(json_file) as f:
            self.json_db = json.load(f)
            self.json_db = self.json_db['companies']
        self.random_seed = random_seed

        self.labels_companies_approx = [1 if np.mean([sample['labels'][0] > 0 for sample in aziend['samples']]) > 0.5 else 0 for aziend in self.json_db]

        self.train_indices, self.test_indices = [], []

        # the shuffle split can have repeated or missing values!
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
        self.paths_labels_train = self.split_abr_mol_all_imgs(mode='train', split=split)
        self.paths_labels_test = self.split_abr_mol_all_imgs(mode='test', split=split)

        return self.paths_labels_train, self.paths_labels_test

    def split_abr_mol_all_imgs(self, mode='train', split=0):
        # iterate over the companies in the train set
        paths_images = []
        labels = []
        dates = []
        companies_cod = []
        paths_images_neigh = []
        labels_neigh = []
        latlong = []
        latlong_neigh = []
        if mode == 'train':
            indices = self.train_indices[split]
        else:
            indices = self.test_indices[split]
        for i in indices:
            for idx, sample in enumerate(self.json_db[i]['samples']):
                if len(sample['imgs']) == 0:
                    msg = f"WARNING: No images for this sample! Date: {sample['date']}"
                    print('-' * (len(msg) + 4))
                    print('= ' + msg + ' =')
                    print('-' * (len(msg) + 4))
                    continue

                imgs_neighbours = [list(x) for x in zip(*sample['imgs_neighbours'])]
                labels_neighbours = [list(x) for x in zip(*sample['labels_neighbours'])][0]
                if mode == "train":
                    for index, s in enumerate(sample['imgs']):
                        paths_images.append(s)
                        labels.append(sample['labels'])
                        dates.append(sample['date'])
                        companies_cod.append(self.json_db[i]['company_cod'])
                        paths_images_neigh.append(imgs_neighbours[index])
                        labels_neigh.append(labels_neighbours)
                        latlong.append((self.json_db[i]['latitude'], self.json_db[i]['longitude']))
                        latlong_neigh.append(sample['latlong_neighbours'])
                else:
                    paths_images.append(sample['imgs'][-1])
                    labels.append(sample['labels'])
                    dates.append(sample['date'])
                    companies_cod.append(self.json_db[i]['company_cod'])
                    paths_images_neigh.append(imgs_neighbours[-1])
                    labels_neigh.append(labels_neighbours)
                    latlong.append((self.json_db[i]['latitude'], self.json_db[i]['longitude']))
                    latlong_neigh.append(sample['latlong_neighbours'])

        latlong = np.array(latlong)
        return list(zip(paths_images, labels, dates, companies_cod, paths_images_neigh, labels_neigh, latlong, latlong_neigh))