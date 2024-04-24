# Spotting Culex pipiens from Satellite: modeling environmental suitability in central Italy with Sentinel-2 and Deep Learning methods

## Data

The code is provided with a single example data point. 

The full dataset is available upon request.

### Pre-processing

**Requires the *Excel* file containing the annotations for the images**. The file can be obtained upon request.

 * First download and convert the Sentinel-2 following [this repository](https://github.com/IZSAM-StatGIS/spotting_cp_satellite)
 * The `.tif` images can be converted in the full dataset using `preprocess/01_CREATE_TORCH_DATASET.py`
 * Use `preprocess/02a_CREATE_DB_MULTI_TEMPORAL.py` or `preprocess/02b_CREATE_DB_GRAPH.py` to obtain the `.json` spec file for the baseline/multi-temporal and MAGAT model respectively.

### Train multi-temporal model
    python main.py --mode=temporal

### Train graph model
    python main.py --mode=graph

### Train baseline (single image) model
    python main.py --mode=base