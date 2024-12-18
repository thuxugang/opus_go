import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model import OPUSGO
from utils import DataGenerator
import numpy as np

import tensorflow as tf
import warnings
import pickle
import horovod.tensorflow as hvd

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch

def main():
    # Initialize Horovod
    hvd.init()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print("rank: ", hvd.local_rank(), "binding ", gpus[hvd.local_rank():hvd.local_rank()+1])
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank():hvd.local_rank()+1], 'GPU')

    params = {}
    params["batch_size"] = 1
    params["d_out"] = 538
    params["d_feat"] = 1280
    params["data_list"] = "/work/home/xugang/projects/weak/go/data/EC_split_seq_embedding_label.pkl"
    params["esm_dir"] = "/work/home/xugang/projects/weak/go/data/esm_feat_ec"
    params["save_path"] = r'./models'

    data_test_gen = DataGenerator(batch_size=1,
                                  data_set="test",
                                  params=params,
                                  rank=0,
                                  size=1)

    my_model = OPUSGO(n_layers=2, 
                      d_model=params["d_feat"], 
                      n_heads=16, 
                      d_ffn=params["d_feat"]*2,
                      d_out=params["d_out"])

    init_len = 256
    my_model(inputs=np.zeros((1, init_len, params["d_feat"]), dtype=np.float32), 
             label=np.zeros((1, params["d_out"]), dtype=np.float32),
             training=False)

    my_model.load_model(name="prot/ec_best_weight.h5")
    
    original_result = []
    test_preds = []
    test_labels = []
    for step, item in enumerate(data_test_gen):

        name, esm_feat, label = item

        out = my_model(esm_feat, label, training=False)
        y_pred = np.max(out, 1)
        y_true = label.numpy()
        
        assert y_pred.shape == y_true.shape
        item = {}
        test_preds.append(torch.from_numpy(y_pred))
        test_labels.append(torch.from_numpy(y_true))
        
        name = name.numpy()[0].decode('utf-8')
        item['name'] = name
        item['y_pred'] = y_pred[0]
        item['y_true'] = y_true[0]
        original_result.append(item)            

    with open("ec_number_our_best.pkl", 'wb') as file:
        pickle.dump(original_result, file)   
        
    preds = torch.cat(test_preds, dim=0)
    labels = torch.cat(test_labels, dim=0)
    print (preds.shape)
    print (labels.shape)
    
    fmax_2 = f1_max(labels, preds)
    aupr_2 = calculate_aupr_fmax_Micro(preds.flatten(), labels.long().flatten())
    print (fmax_2, aupr_2)
    
def calculate_aupr_fmax_Micro(pred, target):
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc

def f1_max(target, pred):
    
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / (is_start.cumsum(0) + 1e-10)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()
        
if __name__ == '__main__':
    print('Running training through horovod.run')
    main()
    