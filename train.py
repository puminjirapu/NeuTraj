from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os
import torch
import time

if __name__ == '__main__':
    print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(config.config_to_str())
    torch.cuda.empty_cache()
    CUDA_LAUNCH_BLOCKING=1
    start = time.time()
    trajrnn = NeuTrajTrainer(tagset_size = config.d, batch_size = config.batch_size,
                             sampling_num = config.sampling_num)    #(128,20,10)
    trajrnn.data_prepare(griddatapath = config.gridxypath, coordatapath = config.corrdatapath,
                         distancepath = config.distancepath, train_radio = config.seeds_radio)
    load_model_name = None
    load_optimizer_name = None
    
    #trajrnn.trained_model_eval(load_model = load_model_name) # for testing
    trajrnn.neutraj_train(load_model = load_model_name,load_optimizer= load_optimizer_name, in_cell_update=config.incell,stard_LSTM=config.stard_unit)   # for training
    
    end = time.time()
    print("training time:",end-start)

