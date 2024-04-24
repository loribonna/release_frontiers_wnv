import pandas as pd
import numpy as np
from pathlib import Path
import json
import datetime
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

    a = np.sin(diff_lat/2) ** 2 + \
        np.cos(lat[:, None]) * np.cos(lat) * np.sin(diff_lng/2)**2

    distances2 = 2 * R * np.arcsin(np.sqrt(a))
    return distances2


def check_neighbours(file_excel):
    """
    Check the neighbours and the min and max distances
    :param file_excel: excel file
    :return:
    """
    # 1- Read excel file
    df = pd.read_excel(file_excel)
    assert 'LAT' in df.columns and 'LONG' in df.columns and 'REGIONE' in df.columns, df.columns

    # 2- unique cod companies
    cod_azienda = df.COD_AZIEND.unique()

    # 3- group by companie and calculate the all2all haversine matrix
    new_df = df.groupby("COD_AZIEND").first()
    latlong = new_df[['LAT', 'LONG']].values
    dist = haversine_fn(latlong=latlong)

    # select the top k nearest neighbours
    indexes = dist.argsort(axis=1)[:, :6]

    # delete self loop
    indexes = indexes[:, 1:]

    # define the Dataframe of the neighbours, save in a file and exploit it in the json creation fn
    df_neigh = pd.DataFrame(
        {cod_azienda[i]: cod_azienda[id] for i, id in enumerate(indexes)}).T

    # print some statistics
    dist_neighbours = dist[np.arange(len(dist))[:, None], indexes]
    print(f"We looking for 5 neighbours.")
    for n in range(5):
        print(
            f"Neighbours {n} --> Min distance over all dataset: {dist_neighbours[:, n].min()}, {dist_neighbours[:, n].max()}")

    # save the neighbours in a file
    neigh = df_neigh.apply(lambda x: [",".join(x)], axis=1)
    df = df.assign(Neighbours="")
    for cod in cod_azienda:
        tmp = df.Neighbours.copy()
        tmp[df.COD_AZIEND == cod] = neigh[cod].copy()
        df.Neighbours = tmp

    fname = f"annotation_dataset_with_6_neighbours.xlsx"
    print("Added neighbours to excel dataset..")
    df.to_excel(fname)
    print(fname)


def create_neighbours_json_dataset(file_excel, torch_dataset_path, label_threshold=0):
    """
    Create the json dataset with the neighbours for the graph approach
    :param file_excel: original excel file with label and info
    :param torch_dataset_path: new path on the nas on which saving the dataset
    :param label_threshold: threshold for the binary label (-1 for raw labels)
    :return:
    """
    # 1- Read excel file
    df = pd.read_excel(file_excel)
    if df.DATA_PRELIEVO.dtype == 'object':
        df.DATA_PRELIEVO = pd.to_datetime(df.DATA_PRELIEVO, format='%d/%m/%Y')

    df = df.sort_values("DATA_PRELIEVO").iloc[::-1]
    assert 'LAT' in df.columns and 'LONG' in df.columns and 'REGIONE' in df.columns

    # 2- unique cod companies
    cod_azienda = df.COD_AZIEND.unique()
    # 3- create json dataset
    json_data = {}
    json_data['companies'] = []

    # 4- define sub dataframe for lat long
    cols = ['COD_AZIEND', 'LAT', 'LONG']
    df_latlong = df.loc[:, cols].set_index("COD_AZIEND")
    errors = []
    fileerr = []

    # 5- populate the json dataset one company at a time
    for id, cod in enumerate(cod_azienda):
        print("processing company: ", cod)

        # take entries of a certain factory
        df_cod = df[df['COD_AZIEND'] == cod]
        df_cod = df_cod.iloc[::-1]
        df_cod.drop_duplicates(subset=['DATA_PRELIEVO'], inplace=True)
        df_cod.reset_index(inplace=True)
        codes_neighbours = df_cod['Neighbours'].iloc[0].split(",")

        samples_list = []
        cod_pos_label1 = 0
        for idx, row in tqdm(enumerate(df_cod.itertuples(index=False, name='Pandas')), total=len(df_cod)):
            # latlong neighbours
            latlong = np.array(
                [df_latlong.loc[c].iloc[0, :].values if df_latlong.loc[c].ndim == 2 else df_latlong.loc[c].values for
                 c in codes_neighbours])
            # date
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

            if idx == 0:
                path_imgs_sample = [
                    str(d) for d in path_imgs_cod if d.parts[-1] <= sample_date]
            else:
                past_date = df_cod.loc[idx-1,
                                       'DATA_PRELIEVO'].strftime("%Y%m%d")

                # if sample_date - past_date there is an high probs that no image occurs
                if (datetime.datetime.strptime(sample_date, "%Y%m%d") - datetime.datetime.strptime(past_date, "%Y%m%d")).days < 5:
                    path_imgs_sample = [
                        [str(d) for d in path_imgs_cod if d.parts[-1] <= sample_date][-1]]
                else:
                    path_imgs_sample = [
                        str(d) for d in path_imgs_cod if past_date <= d.parts[-1] <= sample_date]

            # check if there are images for this sample
            if len(path_imgs_sample) == 0:
                err = {'sample_date': sample_date, 'idx': idx, 'cod': cod}
                msg = f"WARNING: No images for this sample!"
                print('-' * (len(msg) + 4))
                print('= ' + msg + ' =')
                print(f"Cod: {cod}")
                print(f"Sample date: {sample_date}")
                print(f"IDX: {idx}")
                if 'nota' in df_cod.columns:
                    err['aggiuntivo'] = row.nota
                    print(f"IS AGGIUNTIVO: {row.nota}")
                if idx > 0:
                    err['past_date'] = past_date
                    print(f"Past_date {past_date}")
                print('-' * (len(msg) + 4))
                fileerr.append(err)
                continue

            # Fixes sorting of the images (avoid bugs due to different sorting of the files in linux)
            imgs_order = np.argsort([datetime.datetime.strptime(p.split('/')[-1], '%Y%m%d') -
                                     datetime.datetime.strptime(sample_date, '%Y%m%d') for p in path_imgs_sample])
            path_imgs_sample = np.array(path_imgs_sample)[imgs_order].tolist()

            path_imgs_sample_neigh = []
            labels_neigh = []
            skip_neigh = 0
            # for each neighbour
            for id, c in enumerate(codes_neighbours):
                region_neighbour = df[df['COD_AZIEND'] == c].REGIONE.iloc[0]

                # labels of the neighbours
                index_neighbour = abs(
                    df[df['COD_AZIEND'] == c].DATA_PRELIEVO - row.DATA_PRELIEVO).idxmin()
                label_neigh = int(df.loc[index_neighbour, "Culex_pipiens"])
                label_neigh = [
                    1 if label_neigh > label_threshold else 0] if label_threshold >= 0 else [label_neigh]

                # Image paths of the neighbours
                cod_path_neigh = Path(torch_dataset_path) / \
                    region_neighbour / c
                path_imgs_cod_neigh = [
                    d for d in cod_path_neigh.iterdir() if d.is_dir()]

                if idx == 0:
                    path_imgs_cod_neigh = [
                        str(d) for d in path_imgs_cod_neigh if d.parts[-1] <= sample_date]
                else:
                    path_imgs_cod_neigh = [
                        str(d) for d in path_imgs_cod_neigh if past_date <= d.parts[-1] <= sample_date]

                # Fixes sorting of the images (avoid bugs due to different sorting of the files in linux)
                imgs_order = np.argsort([datetime.datetime.strptime(p.split('/')[-1], '%Y%m%d') -
                                         datetime.datetime.strptime(sample_date, '%Y%m%d') for p in path_imgs_cod_neigh])
                path_imgs_cod_neigh = np.array(path_imgs_cod_neigh)[
                    imgs_order].tolist()

                # when the neighbour present only future data the list is empty
                # also, if the neighbour present a date to far from the current one, discard the image
                if not path_imgs_cod_neigh:
                    # delete the coordinates corresponding to the empty neighbours
                    latlong = np.delete(latlong, id-skip_neigh, axis=0)
                    skip_neigh += 1
                    continue
                elif abs(datetime.datetime.strptime(Path(path_imgs_cod_neigh[-1]).parts[-1], "%Y%m%d") - row.DATA_PRELIEVO).days <= 50:
                    path_imgs_sample_neigh.append(path_imgs_cod_neigh)
                    labels_neigh.append(label_neigh)
                else:
                    latlong = np.delete(latlong, id - skip_neigh, axis=0)
                    skip_neigh += 1
                    continue

            # keep a number of sample equal to the minimum number of valid images between neigh and the current
            if len(path_imgs_sample_neigh) == 0:
                if idx > 0:
                    errors.append((cod, idx, past_date, sample_date))
                else:
                    errors.append((cod, idx, -1, sample_date))
                continue

            min_len = min(
                min([len(p) for p in path_imgs_sample_neigh]), len(path_imgs_sample))
            # keep closes images by date
            path_imgs_sample = path_imgs_sample[-min_len:]
            path_imgs_sample_neigh = [p[-min_len:]
                                      for p in path_imgs_sample_neigh]

            # Keep max 5 neighbours
            N_NEIGHS = 5
            while len(path_imgs_sample_neigh) < N_NEIGHS:
                path_imgs_sample_neigh.append(path_imgs_sample_neigh[-1])
                labels_neigh.append(labels_neigh[-1])
                latlong = np.append(
                    latlong, np.atleast_2d(latlong[-1]), axis=0)

            if len(path_imgs_sample_neigh) > N_NEIGHS:
                path_imgs_sample_neigh = path_imgs_sample_neigh[:N_NEIGHS]
                labels_neigh = labels_neigh[:N_NEIGHS]
                latlong = latlong[:N_NEIGHS]

            # append to the sample list
            samples_list.append(
                {
                    "date": sample_date,
                    "labels": labels,
                    "imgs": path_imgs_sample,
                    "imgs_neighbours": path_imgs_sample_neigh,
                    "labels_neighbours": labels_neigh,
                    "latlong_neighbours": latlong.tolist()
                })
        print(
            f"Date with no corresponding images: {np.array([1 for s in samples_list if len(s['imgs']) == 0]).sum()}")
        global_mean_label = cod_pos_label1 / len(samples_list)
        # append the dictionary of the entire data for a company
        json_data['companies'].append(
            {
                "company_cod": cod,
                "global_mean_label": global_mean_label,
                "latitude": df_cod.LAT.values[0],
                "longitude": df_cod.LONG.values[0],
                "region": row.REGIONE,
                "samples": samples_list
            })

    print("end..write on json file!")
    print("Errors: ", len(errors))
    print(errors)
    fname = 'graph_base.json'
    with open(fname, 'w') as outfile:
        json.dump(json_data, outfile, indent=2)
    print(fname)

    print('-----')
    print("Errors:\n", len(fileerr))
    pd.DataFrame(fileerr).to_csv(f'errors_graph_dataset.csv', index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--torch_dataset_path', type=str,
                        help="Path of the processed torch dataset")
    parser.add_argument('--file_excel', type=str, required=True,
                        help="Excel file with the annotations for the dataset")
    parser.add_argument('--label_threshold', type=int, default=0,
                        help="Threshold for pipiens capture label. -1 for RAW labels.")

    args = parser.parse_args()

    check_neighbours(file_excel=args.file_excel)

    create_neighbours_json_dataset(file_excel=args.file_excel, torch_dataset_path=args.torch_dataset_path, label_threshold=args.label_threshold,
                                   multilabel=False)


if __name__ == '__main__':
    main()
