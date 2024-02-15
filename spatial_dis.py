import multiprocessing
import traj_dist.distance as tdist
from tools import config
import pickle
import random
import numpy as np
import numba
import os
import time

def com_true_test(path = None):
    test_set = pickle.load(open(path, 'rb'),encoding="latin1")
    print("test_set",len(test_set))
    # test_set, _, _ = pickle.load(open(path, 'rb'))
    test_set = test_set[1800:6000]
    test_sample = test_set[:600]

    process_size = 200 
    pool = multiprocessing.Pool(processes=20)
    print("test_sample:",len(test_sample))
    for i in range(len(test_sample)+1):
        if (i!=0) and (i%process_size==0):
            pool.apply_async(com_true_test_batch,(i,test_sample[(i-process_size):i],test_set))
    pool.close()
    pool.join()

def com_true_test_batch(i,test_sample,test_set):  # 200*30w  50 file
    test_sample = np.array(test_sample)
    test_set = np.array(test_set)

    batch_matrix = tdist.cdist(test_sample,test_set,metric=config.distance_type)
    batch_matrix = np.array(batch_matrix)
    filename = './features/{}/test_ground_truth/{}/spatial_truth/{}_{}_{}.npy'.format(config.data_type,config.distance_type, config.data_type, config.distance_type, str(i))
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    pickle.dump(batch_matrix,open(filename,'wb'))
    np.save('./features/{}/test_ground_truth/{}/spatial_truth/{}_{}_{}.npy'.format(config.data_type,config.distance_type, config.data_type, config.distance_type, str(i)), batch_matrix)

    print('complete: '+str(i))

def combine_true_test_batch():
    all_batch = []
    for i in range(600): 
        if (i!=0) and (i%200==0):
            all_batch.append(np.load('./features/{}/test_ground_truth/{}/spatial_truth/{}_{}_{}.npy'.format(config.data_type,config.distance_type, config.data_type, config.distance_type, str(i)), allow_pickle=True))
            print(i)

    all_batch = np.concatenate(all_batch, axis=0)
    print(all_batch.shape)

    filename = './features/{}/test_ground_truth/{}/spatial_truth/{}_{}.npy'.format(config.data_type,config.distance_type, config.data_type, config.distance_type) 
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    pickle.dump(all_batch,open(filename,'wb'))

def merge_ST(s_path=None):# dlhu, t_path=None):
    s = np.load(s_path,allow_pickle=True)
    s = s/np.max(s)
    st = s

    filename = './features/{}/test_ground_truth/{}/{}_{}_st.npy'.format(config.data_type, config.distance_type, config.data_type, config.distance_type)
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    pickle.dump(st,open(filename,'wb'))
 
if __name__ == '__main__':
    start = time.time()
    com_true_test(path= './data/toy_trajs')
    combine_true_test_batch()
    merge_ST(s_path='./features/{}/test_ground_truth/{}/spatial_truth/{}_{}.npy'.format(config.data_type,config.distance_type, config.data_type, config.distance_type))
    end = time.time()
    print("spatial_dis time:",end-start)
