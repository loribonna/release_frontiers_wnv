# Spotting Culex pipiens from Satellite: modeling environmental suitability in central Italy with Sentinel-2 and Deep Learning methods

## Data

The code is provided with a single example data point. 

The full dataset is available upon request.

### Train multi-temporal model
    python main.py --mode=temporal

### Train graph model
    python main.py --mode=graph

### Train baseline (single image) model
    python main.py --mode=base