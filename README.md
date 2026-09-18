## DGPD

This is the PyTorch implementation for our manuscript submitted to the *IEEE Internet of Things Journal*:

### Environment Requirement

Python 3.10 or 3.11 is recommended. The required packages are as follows:

- PyTorch >= 2.3
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0
- tqdm >= 4.65
- haversine >= 2.8

Install the dependencies with:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU experiments, install the PyTorch build compatible with the CUDA version on your machine.


To preprocess a TSMC2014/Foursquare NYC or TKY file, first specify `RAW_DATA_PATH` and `DATASET_NAME` at the beginning of `pro.py`, and then run:

```shell
python pro.py
```
### Running Example

To conduct an experiment on `Foursquare-Tokyo`, run:

```shell
python main.py --dataset tky --device cuda --batch_size 32 --num_epochs 30
```

The code chronologically splits each user's trajectory into 80% training, 10% validation history, and 10% testing. Temporal and geographical relation matrices are generated automatically during the first run. 


Please cite our paper if you use this code.
