import tools.test_methods as tm
import time, os, pickle
import numpy as np
import torch
from tools import  config
from tools import sampling_methods as sm
from geo_rnns.neutraj_model import NeuTraj_Network
from geo_rnns.wrloss import WeightedRankingLoss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

def pad_sequence(traj_grids, maxlen=100, pad_value = 0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0])*pad_value
        while (len(traj) < maxlen):    # 对于没到最大轨迹长度的坐标用0补齐
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs

class NeuTrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate = config.learning_rate):

        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self, griddatapath = config.gridxypath,
                     coordatapath = config.corrdatapath,
                     distancepath = config.distancepath,
                     train_radio = config.seeds_radio):
        dataset_length = config.datalength   # 1800
        traj_grids, useful_grids, max_len = pickle.load(open(griddatapath, 'rb'))
        self.trajs_length = [len(j) for j in traj_grids][:10000] #10000 # 取前1800条轨迹的轨迹长度
        self.grid_size = config.gird_size  #[1100, 1100]
        self.max_length = max_len
        grid_trajs = [[[i[0]+config.spatial_width , i[1]+config.spatial_width] for i in tg]  # 每个轨迹点的grid index长宽都扩大2
                      for tg in traj_grids[:10000]] #10000 #dlhu #14000

        traj_grids, useful_grids, max_len = pickle.load(open(coordatapath, 'rb'))
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)   # 平均值和全局标准差
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy] for r in t] for t in traj_grids] # 对轨迹点坐标进行归一化

        coor_trajs = traj_grids[:10000] #10000 #dlhu #14000
        # train_size = int(len(grid_trajs)*train_radio/self.batch_size)*self.batch_size  # 训练集大小360--------------------------
        train_size = 3000 #3000 #10000  # ----------------------------

        grid_train_seqs, grid_test_seqs = grid_trajs[:train_size], grid_trajs[train_size:]  # 划分测试集和训练集
        coor_train_seqs, coor_test_seqs = coor_trajs[:train_size], coor_trajs[train_size:]

        self.grid_trajs = grid_trajs
        self.grid_train_seqs = grid_train_seqs
        self.coor_trajs = coor_trajs
        self.coor_train_seqs = coor_train_seqs
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append([coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)  # 将coor和grid整合到一起

        print("Padded Trajs shape")
        print(len(pad_trjs))   # 1800
        print("max_len:",max_len) #dlhu
        self.train_seqs = pad_trjs[:train_size]
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen= max_len))   # 这里对1800条轨迹都用过了pad_sequence算法，对轨迹进行补全，那么后续应该就不需要再调用pad_sequence算法了
        #with open(distancepath, 'rb') as f:
         #   distance = pickle.load(open(f),'rb')
        distance = pickle.load(open(distancepath,'rb'))
        # distance = np.load(distancepath)
        max_dis = distance.max()
        print('max value in distance matrix :{}'.format(max_dis))
        print(config.distance_type)
        if config.distance_type == 'dtw':
            distance = distance/max_dis
        print("Distance shape")
        print(distance[:train_size].shape) #dlhu(3000,10000)  # (360,1874)
        train_distance = distance[:train_size, :train_size] #dlhu (3000,3000) # (360,360)

        print("Train Distance shape")
        print(train_distance.shape)
        self.distance = distance
        self.train_distance = train_distance

    def batch_generator(self, train_seqs, train_distance): # (360,360)个测试集和ground-truth集
        j = 0
        while j< len(train_seqs):
            anchor_input, trajs_input, negative_input,distance,negative_distance = [],[],[],[],[]
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            for i in range(self.batch_size):  #
                # sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)   # --------------------
                # negative_sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)   # --------------------

                sampling_index_list = sm.distance_sampling(self.distance,len(self.train_seqs), j + i) #采10个正样本
                negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i) #采10个负样本

                trajs_input.append(train_seqs[j+i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                if j+i not in batch_trajs_keys:
                    batch_trajs_keys[j+i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j+i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(np.exp(-float(train_distance[j+i][traj_index])*config.mail_pre_degree))

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            #normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)

            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length) # 前面已经对所有1800条轨迹进行pad_sequence算法操作过了，这里的pad_sequence应该是多余的吧
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))


            yield ([np.array(anchor_input),np.array(trajs_input),np.array(negative_input), np.array(batch_trajs_input)],  # 生成器，每次调用都从上一次中断的地方继续执行
                   [anchor_input_len, trajs_input_len, negative_input_len, batch_trajs_len],
                   [np.array(distance),np.array(negative_distance)])
            j = j + self.batch_size


    def trained_model_eval(self, print_batch = 10 ,print_test = 100,save_model = True, load_model = None,load_optimizer=None,
                           in_cell_update = True, stard_LSTM = False):

        spatial_net = NeuTraj_Network(4, self.target_size, self.grid_size,
                                      self.batch_size, self.sampling_num,
                                      stard_LSTM= stard_LSTM, incell= in_cell_update)

        if load_model != None:
            spatial_net.load_state_dict(pickle.load(open(load_model, 'rb'))) #(torch.load(load_model))

            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)
            acc1 = tm.test_model(self,embeddings, test_range=list(range(10000)), #dlhu #16000
                                 similarity=True, print_batch=print_test, r10in50=True)
            print(acc1)
            return acc1


    def neutraj_train(self, print_batch = 10, print_test = 100, save_model = True, load_model = None,load_optimizer=None,
                      in_cell_update= True, stard_LSTM = False):

        spatial_net = NeuTraj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM= stard_LSTM, incell= in_cell_update)

        optimizer = torch.optim.Adam([p for p in spatial_net.parameters() if p.requires_grad], lr=config.learning_rate)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num)

        spatial_net.cuda()
        mse_loss_m.cuda()

        lastepoch = '0'
        if load_model != None:
            spatial_net.load_state_dict(torch.load(load_model))  # 加载save的模型参数，并赋给网络
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch=load_model.split('_')[2]     # 提取出上一个epoch的值

            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch)   # embedding的范围需要在test_methods.py中按需修改

            tm.test_model(self,embeddings, test_range=list(range(12000)), #dlhu #16000
                                 similarity=True, print_batch=print_test, r10in50=True)

        maxacc = 0.0
        maxepoch = 0
        for epoch in range(int(lastepoch),config.epochs):
            spatial_net.train()
            print("Start training Epochs : {}".format(epoch))
            # print len(torch.nonzero(spatial_net.rnn.cell.spatial_embedding))
            start = time.time()

            for i, batch in enumerate(self.batch_generator(self.train_seqs, self.train_distance)):

                inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]  # 采样到的输入轨迹、输入轨迹的长度、相似度ground-truth

                print("inputs_arrays:",inputs_arrays[0].shape, inputs_arrays[1].shape, inputs_arrays[2].shape, inputs_arrays[3].shape, inputs_arrays) #dlhu

                trajs_loss, negative_loss = spatial_net(inputs_arrays, inputs_len_arrays)  # 返回anchor与positive和negative嵌入之后的相似度距离

                positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))
                negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))

                loss = mse_loss_m(trajs_loss,positive_distance_target,negative_loss,negative_distance_target) # 传入embedding后的距离，和ground-truth距离,计算损失

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                optim_time = time.time()
                if not in_cell_update:
                    spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)
                batch_end = time.time()

            end = time.time()

            #dlhu
            print('Epoch [{}/{}], Step [{}/{}], Positive_Loss: {}, Negative_Loss: {}, Total_Loss: {}, ' \
                    'Time_cost: {}'. \
                    format(epoch, config.epochs, i, len(self.train_seqs),self.batch_size, mse_loss_m.trajs_mse_loss.item(), mse_loss_m.negative_mse_loss.item(),loss.item(), end - start))

            embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch= config.em_batch) # 得到轨迹的embedding
            print('len(embeddings): {}'.format(len(embeddings)))
            print(embeddings.shape)
            print(embeddings[0].shape)

            acc1 = tm.test_model(self,embeddings, test_range=list(range(3000,10000)),similarity=True, print_batch=print_test) #dlhu #(10000,14000)

            print(acc1)

            if save_model and epoch>0 and epoch%50==0:
                save_model_name = './model/{}/{}/{}_epoch_{}' \
                                      .format(config.data_type,config.distance_type, config.data_type, str(epoch)) + \
                                  '_HR10_{}_HR50_{}_HR1050_{}_LOSS_{}.pkl'.format(acc1[0], acc1[1], acc1[2],loss.item())
                print(save_model_name)
                os.makedirs(os.path.dirname(save_model_name),exist_ok=True)
                pickle.dump(spatial_net.state_dict(),open(save_model_name,'wb'))
                #torch.save(spatial_net.state_dict(), save_model_name)
                
                save_optimizer_name = './optimizer/{}/{}/{}_epoch_{}.pkl'.format(config.data_type, config.distance_type, config.data_type,str(epoch))
                os.makedirs(os.path.dirname(save_optimizer_name),exist_ok=True)
                pickle.dump(optimizer.state_dict(),open(save_optimizer_name,'wb'))
                #torch.save(optimizer.state_dict(), save_optimizer_name)
