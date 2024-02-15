from tools import  config
import numpy as np
import torch.autograd as autograd
import torch
# from geo_rnns.spatial_memory_lstm_pytorch import SpatialCoordinateRNNPytorch
import time
import csv
import codecs
import pickle


def test_comput_embeddings(self, spatial_net, test_batch = 1025):
    if config.recurrent_unit=='GRU' or config.recurrent_unit=='SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    embeddings_list = []
    j = 0
    s = time.time()

    while j < self.padded_trajs.shape[0]:

        # for i in range(self.batch_size):
        out = spatial_net.rnn([autograd.Variable(torch.Tensor(self.padded_trajs[j:j+test_batch]),
                                                 requires_grad=False).cuda(),
                               self.trajs_length[j:j+test_batch]],hidden)
        # embeddings = out.data.cpu().numpy()
        embeddings = out.data
        j += test_batch
        embeddings_list.append(embeddings)
        if (j% 600) == 0:
            print(j)
    print(('embedding time of {} trajectories: {}'.format(self.padded_trajs.shape[0], time.time()-s)))
    embeddings_list = torch.cat(embeddings_list, dim=0)
    print((embeddings_list.size()))
    return embeddings_list.cpu().numpy()

def test_model(self, embedding_set, test_range, print_batch=10, similarity = False, r10in50 = False):

    input_dis_matrix = np.load('./features/{}/test_ground_truth/{}/{}_{}_st.npy'.format(config.data_type, config.distance_type, config.data_type, config.distance_type), allow_pickle=True)
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for i, t in enumerate(embedding_set): 
        if i == 4200:
            break
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb - embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0

    for i in range(len(input_dis_matrix)):

        input_r = np.array(input_dis_matrix[i])[:4200]
        input_r50 = np.argsort(input_r)[1:51] 
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])[:4200]
        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10)))) 
        l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
        l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))

    recall_10 = float(l_recall_10) / (10 * len(input_dis_matrix))
    recall_50 = float(l_recall_50) / (50 * len(input_dis_matrix))
    recall_10_50 = float(l_recall_10_50) / (10 * len(input_dis_matrix))

    print(recall_10, recall_50, recall_10_50)

    return recall_10, recall_50, recall_10_50
   

if __name__ == '__main__':
    print((config.config_to_str()))
