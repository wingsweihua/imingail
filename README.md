# ImInGAIL: Learning to Simulate on Sparse Trajectory Data (ECML/PKDD'20)



```
@inproceedings{imingail,
 author = {Wei, Hua and Chen, chacha and Liu, Chang and Zheng, Guanjie and Li, Zhenhui},
 title = {Learning to Simulate on Sparse Trajectory Data},
 booktitle = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
 series = {ECML/PKDD '20},
 year = {2020},
 organization={Springer}
} 
```

Usage and more information can be found below.

## Usage

How to run the code:

The code relies on the simulator of CityFlow, which provides scalable, multi-process simulation for transportation sceanrios.

1. Intall the gym environment 'gym_citycar-v0' for CityFlow in your python environment.

    ``https://github.com/wingsweihua/gym-citycar.git``

    There are two branches, one for intersection network and one for ring network.
 
2. Pull the codes for ImInGAIL.

    ``git clone https://github.com/wingsweihua/imingail.git``
    
Start an experiment in the python environment with 'gym_citycar-v0' with the following script:

    ``bash runexp_gail_sparse.sh``
    


## Dataset
* synthetic data

  - simulation file: Traffic file and road networks can be found in ``data/1_1`` && ``data/ring``.
  - logged data: ``data/expert_trajs/1_1`` && ``data/expert_trajs/ring``

* real-world data

  - simulation file: Traffic file and road networks can be found in ``data/1x4_LA`` && ``data/4x4_gudang``.
  - logged data: ``data/expert_trajs/1_4`` && ``data/expert_trajs/4_4``