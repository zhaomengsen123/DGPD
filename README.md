## DGPD

This is the PyTorch implementation for our manuscript submitted to the *IEEE Internet of Things Journal*:

> Sai Zhao, Chunxiao Li, Caisen Chen, and Shuai He.  
> **Dual-Graph Prior-Guided Decoupled Debiasing for Next POI Recommendation.**

The bibliographic information below can be updated after publication:

```bibtex
@article{zhao_dgpd,
  author  = {Zhao, Sai and Li, Chunxiao and Chen, Caisen and He, Shuai},
  title   = {Dual-Graph Prior-Guided Decoupled Debiasing for Next POI Recommendation},
  journal = {IEEE Internet of Things Journal},
  note    = {Under review}
}
```

In this paper, we propose DGPD, a dual-graph prior-guided decoupled debiasing framework for next point-of-interest (POI) recommendation. DGPD models transition and geographical POI relations through two representation views and employs a spatio-temporal sequence encoder to capture users' dynamic mobility preferences.

The implementation further derives a mobility-aware transition prior from collective check-in transitions and a recency-aware revisit prior from individual historical trajectories. These priors calibrate the neural prediction scores to reduce the effects of popularity, transition sparsity, and local behavioral noise. 

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
