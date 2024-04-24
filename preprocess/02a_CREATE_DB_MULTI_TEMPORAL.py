import pandas as pd
import numpy as np
from pathlib import Path
import json
import datetime
import json
import argparse


def create_json_dataset(file_excel, torch_dataset_path, label_threshold=0):
    """
    Create dataset without neighbours
    :param file_excel: original file excel
    :param torch_dataset_path: path of the pre-processed torch dataset
    :param label_threshold: threshold for the binary label (-1 for raw labels)
    :return:
    """
    # 1- Read excel file
    df = pd.read_excel(file_excel)

    if df.DATA_PRELIEVO.dtype == 'object':
        df.DATA_PRELIEVO = pd.to_datetime(df.DATA_PRELIEVO, format='%d/%m/%Y')

    # 2- unique cod companies
    cod_azienda = df.COD_AZIEND.unique()

    # 3- create json dataset
    json_data = {}
    json_data['companies'] = []

    # 4- populate the json dataset one company at a time
    for cod in cod_azienda:
        print("processing company: ", cod)

        # take entries of a certain factory
        df_cod = df[df['COD_AZIEND'] == cod]
        samples_list = []
        cod_pos_label1 = 0
        for row in df_cod.itertuples(index=False, name='Pandas'):
            sample_date = row.DATA_PRELIEVO.strftime('%Y%m%d')

            labels = [row.Culex_pipiens]

            # if label_threshold is set to -1, the labels are kept raw
            # else the labels are thresholded to 0 or 1
            labels = [
                1 if lab > label_threshold else 0 for lab in labels] if label_threshold >= 0 else labels
            cod_pos_label1 += labels[0]
            # select the paths of the relative company
            cod_path = Path(torch_dataset_path) / row.REGIONE / cod
            path_imgs_cod = [d for d in cod_path.iterdir() if d.is_dir()]
            path_imgs_sample = [
                str(d) for d in path_imgs_cod if d.parts[-1] <= sample_date]

            # Fixes sorting of the images (avoid bugs due to different sorting of the files in linux)
            imgs_order = np.argsort([datetime.datetime.strptime(p.split('/')[-1], '%Y%m%d') -
                                     datetime.datetime.strptime(sample_date, '%Y%m%d') for p in path_imgs_sample])
            path_imgs_sample = np.array(path_imgs_sample)[imgs_order].tolist()

            # append to the sample list
            samples_list.append(
                {
                    "date": sample_date,
                    "labels": labels,
                    "imgs": path_imgs_sample,
                })
        # compute the global mean label for the company
        global_mean_label = cod_pos_label1 / len(samples_list)

        # append the dictionary of the entire data for a company
        json_data['companies'].append(
            {
                "company_cod": cod,
                "global_mean_label": global_mean_label,
                "latitude": df_cod.LAT.values[0] if 'LAT' in df_cod.columns else None,
                "longitude": df_cod.LONG.values[0] if 'LONG' in df_cod.columns else None,
                "region": row.REGIONE,
                "samples": samples_list
            })
    print("end..\nwrite on json file!")
    fname = 'multitemporal_base.json'
    with open(fname, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
    print(fname)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--torch_dataset_path', type=str,
                        help="Path of the processed torch dataset")
    parser.add_argument('--file_excel', type=str, required=True,
                        help="Excel file with the annotations for the dataset")
    parser.add_argument('--label_threshold', type=int, default=0,
                        help="Threshold for pipiens capture label. -1 for RAW labels.")

    args = parser.parse_args()

    create_json_dataset(args.file_excel, args.torch_dataset_path,
                        label_threshold=args.label_threshold)


if __name__ == '__main__':
    main()
