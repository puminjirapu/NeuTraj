# NeuTraj

This is a seed guided neural metric learning approach for calculating trajectory similarities.

## Require Packages
Pytorch, Numpy, trajectory_distance

## Running Procedures

### Create Folders
Please create 3 empty folders:

*`data`: Path of the original data which is organized to a trajectory list. Each trajectory in it is a list of coordinate tuples (lon, lat). We provide the origin data and the data with noises.

*`features`: This folder contains the features that generated after the preprocessing.py. It contains four files: coor_seq, grid_seq, index_seq and seed_distance. 

*`model`: It is used for placing the NeuTraj model of each training epoch.

### Preprocessing
Run `preprocessing.py`. It filters the original data and maps the coordinates to grids. After such process, intermediate files which contain `coor_seq`, `grid_seq`, and `index_seq` are generated.

### Ground Truth Generation
Run `spatial_dis.py`. It computes the similarity values between training data.

### Training & Evaluating
Run `start_train.py`. It trains NeuTraj under the supervision of seed distance. The parameters of NeuTraj can be modified in /tools/config.py.
