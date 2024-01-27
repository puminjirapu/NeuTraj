# Data path
data_type = 'AIS'
distance_type = 'dtw'

corrdatapath = './features/{}/{}_traj_coord'.format(data_type,data_type)
gridxypath = './features/{}/{}_traj_grid'.format(data_type,data_type)
distancepath = './features/{}/ground_truth/{}_{}'.format(data_type,data_type,distance_type)
# distancepath = './features/{}/test_ground_truth/{}/{}_{}_st.npy'.format(data_type, distance_type, data_type, distance_type)

# Training Prarmeters
GPU = "0"

all_traj_number=10000
learning_rate = 0.01
seeds_radio=0.3
epochs = 100
batch_size = 20
sampling_num = 10



if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8




# Test Config
datalength = 3000 #seed trajectory
em_batch = 100
test_num = 1000

# Model Parameters
d = 128
stard_unit = False # It controls the type of recurrent unit (standrad cells or SAM argumented cells)
incell = True
recurrent_unit = 'LSTM' #GRU, LSTM or SimpleRNN
spatial_width  = 4 

gird_size = [2600, 2600] 


def config_to_str():
   configs = 'learning_rate = {} '.format(learning_rate)+ '\n'+\
             'mail_pre_degree = {} '.format(mail_pre_degree)+ '\n'+\
             'seeds_radio = {} '.format(seeds_radio) + '\n' +\
             'epochs = {} '.format(epochs)+ '\n'+\
             'datapath = {} '.format(corrdatapath) +'\n'+ \
             'datatype = {} '.format(data_type) + '\n' + \
             'corrdatapath = {} '.format(corrdatapath)+ '\n'+ \
             'distancepath = {} '.format(distancepath) + '\n' + \
             'distance_type = {}'.format(distance_type) + '\n' + \
             'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
             'batch_size = {} '.format(batch_size)+ '\n'+\
             'sampling_num = {} '.format(sampling_num)+ '\n'+\
             'incell = {}'.format(incell)+ '\n'+ \
             'stard_unit = {}'.format(stard_unit)
   return configs


if __name__ == '__main__':
    print(('../model/model_training_600_{}_acc_{}'.format((0),1)))
    print((config_to_str()))
