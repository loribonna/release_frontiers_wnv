import os
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import cv2
import torch
from torchvision import transforms
import json
import argparse
from tqdm import tqdm


def nearest(items, pivot, mode):
    if mode == "day":
        init, end = 6, 14
    else:
        init, end = 8, 16
    if min(abs(datetime.datetime.strptime(x.parts[-1][init:end], '%Y%m%d') -
               datetime.datetime.strptime(pivot, "%Y%m%d")) for x in items).days > 60:
        return None
    else:
        return min(items, key=lambda x: abs(datetime.datetime.strptime(x.parts[-1][init:end], '%Y%m%d') -
                                            datetime.datetime.strptime(pivot, "%Y%m%d")))


def custom_loader(path):
    # load the image
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    w, h = img.shape
    return img.reshape(w, h, 1)


def bands_to_torch(single_img_bands, base_path, day_temp=None, night_temp=None, rain_temp=None):
    """
    function that take bands path as input, convert to torch tensor and return the new path on nas
    :param single_img_bands: bands of one image
    :param base_path: path on which to save the torch tensors
    :param day_temp: daily temp relate to the idpoint
    :param night_temp: nightly temp relate to the idpoint
    :return:
    """
    # "[base_path]/dataset_torch/Region/[Cod_company]/[date]/bands.pt"
    # 1- create the correct path to save the images
    if len(single_img_bands[0].parts[-1].split('_')) == 4:
        cod_company = single_img_bands[0].parts[-1].split('_')[-1].split('.')[
            0]
    else:
        cod_company = single_img_bands[0].parts[-1].split(
            '_')[-2]+'_'+single_img_bands[0].parts[-1].split('_')[-1].split('.')[0]
    region = single_img_bands[0].parts[-2]
    date = single_img_bands[0].parts[-1][7:15]
    path_to_save = Path(base_path) / region / cod_company / date

    # 2 - check if the images is already present, if not create the correct directory
    path_to_save.mkdir(parents=True, exist_ok=True)

    # 3 - Read the bands once at a time and convert to a single torch tensor
    bands = ["AOT", "B02", "B03", "B04", "B05", "B06", "B07",
             "B8A", "WVP", "B11", "B12", "CLD", "SNW", "SCL"]
    img_bands = []
    for b in bands:
        path_band = [s for s in single_img_bands if b in str(s)]
        # if a band lacks don't save the relative images
        assert path_band, f"Band {b} not found in {single_img_bands}"

        img_band = custom_loader(path=path_band[0])
        img_bands.append(img_band)

    # add the temperature bands if present
    if day_temp is None or night_temp is None:
        day_temp = np.zeros_like(img_bands[0])
        night_temp = np.zeros_like(img_bands[0])
    else:
        day_temp = custom_loader(path=day_temp)
        day_temp = [(1-0)/(44.71-(-11.03))]*(day_temp-(-11.03)) + 0
        night_temp = custom_loader(path=night_temp)
        night_temp = [(1-0)/(24.27-(-18.33))]*(night_temp-(-18.33)) + 0

    rain_temp = np.zeros_like(img_bands[0])

    img_bands.append(day_temp)
    img_bands.append(night_temp)
    img_bands.append(rain_temp)
    img_bands = np.concatenate(img_bands, axis=2)
    torch_bands = transforms.ToTensor()(img_bands)

    # 4- save the tensor
    torch.save(torch_bands, path_to_save / 'bands.pt')


def create_torch_dataset(input_path, file_excel, output_path, path_temperature_data=None):
    """
    Create the torch version of the dataset for speed up the processing
    :param input_path: original path with .tif images
    :param file_excel: file excel containing labels and image paths
    :param output_path: output path for the torch dataset
    :param path_temperature_data: path for the temperature data
    :return:
    """
    # Read the excel file
    df = pd.read_excel(file_excel)

    # Different folder for different companies
    cod_azienda = df.COD_AZIEND.unique()

    count_empty = 0
    for id, cod in enumerate(cod_azienda):

        # take entries of a certain factory
        region = df[df['COD_AZIEND'] == cod]['REGIONE'].values[0]
        path_imgs_companies = [p for p in (
            Path(input_path) / region).joinpath().glob("*" + cod + "*.tiff")]

        if path_temperature_data is not None:
            modis_night = [p for p in Path(
                path_temperature_data).joinpath().glob("*Night*" + cod + "*.tif")]
            modis_night.extend(
                [p for p in Path(path_temperature_data).joinpath().glob("*LSTN_*" + ".tif")])
            modis_day = [p for p in Path(
                path_temperature_data).joinpath().glob("*Day*" + cod + "*.tif")]
            modis_day.extend(
                [p for p in Path(path_temperature_data).joinpath().glob("*LSTD_*" + ".tif")])
            if not modis_night:
                print(path_temperature_data, cod)
                count_empty += 1
                print(f"empty: {count_empty}")

        dates = [p.parts[-1][7:15] for p in path_imgs_companies]
        dates_datetime = [datetime.datetime.strptime(
            str(date), "%Y%m%d") for date in dates]

        imgs = pd.DataFrame(
            {'imgs': path_imgs_companies, 'dates': dates_datetime})

        for single_img_bands in tqdm(imgs.groupby('dates').imgs.apply(list), desc=f"[{id+1}/{len(cod_azienda)}] {cod}"):
            current_date = single_img_bands[0].parts[-1][7:15]

            if len(modis_day) > 0:
                day_temp = nearest(modis_day, current_date,
                                   "day") if path_temperature_data is not None else None
                night_temp = nearest(modis_night, current_date,
                                     "night") if path_temperature_data is not None else None
            else:
                day_temp = None
                night_temp = None

            bands_to_torch(single_img_bands, output_path,
                           day_temp, night_temp)
    print("Finish dataset torch creation...")


def check_empty_dir(dirName):
    """
    checking for empty dirs
    :param dirName: path to the root dir
    """
    listOfEmptyDirs = [dirpath for (dirpath, dirnames, filenames) in os.walk(dirName) if
                       len(dirnames) == 0 and len(filenames) == 0]
    if len(listOfEmptyDirs) > 0:
        MyFile = open('folder_with_missing_data.txt', 'w')
        for element in listOfEmptyDirs:
            MyFile.write(str(element))
            MyFile.write('\n')
        MyFile.close()

        raise ValueError(f"{len(listOfEmptyDirs)} empty directories found!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', default='./data/abruzzo_zanzare/new_dataset_torch/',
                        type=str, help="Out path for torch dataset")
    parser.add_argument('--file_excel', type=str, required=True,
                        help="Excel file with the annotations for the dataset")
    parser.add_argument('--path_temperature_data', required=False,
                        help="Path for temperature (MODIS) data.")

    args = parser.parse_args()

    check_empty_dir(args.output_path)

    create_torch_dataset(args.input_path, args.file_excel, args.output_path,
                         args.path_temperature_data)


if __name__ == '__main__':
    main()
