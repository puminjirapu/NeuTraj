from tools import preprocess
from tools.distance_compution import trajectory_distance_combain,trajecotry_distance_list
import pickle
import numpy as np
#import torch
from tools import config
import time

def distance_comp(coor_path):
    traj_coord = pickle.load(open(coor_path, 'rb'),encoding="latin1")[0]
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print("len(np_traj_coord):",len(np_traj_coord))

    distance_type = config.distance_type

    trajecotry_distance_list(np_traj_coord[:3000], batch_size=500, processors=20, distance_type=distance_type, data_name=config.data_type)

    trajectory_distance_combain(3000, batch_size=500, metric_type=distance_type, data_name=config.data_type) 

if __name__ == '__main__':
    start = time.time()

    coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/toy_trajs') 
    distance_comp(config.corrdatapath)
    end = time.time()
    print("preprocessing time:",end-start)
