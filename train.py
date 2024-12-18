import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from model import OPUSGO, calculate_metrics
from utils import DataGenerator
import time
import numpy as np
from tensorflow import keras

import tensorflow as tf
import warnings

import horovod.tensorflow as hvd

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
nohup mpirun -np 16 -bind-to none -map-by slot python -u train.py  > log &
"""

def model_train(hvd, esm_feat, label, my_model, optimizer):
    
    with tf.GradientTape() as tape:
        loss, out = my_model(esm_feat, label, training=True)
        
    trainable_variables = my_model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm=1.)
    reduced_grads = hvd.grouped_allreduce(gradients)
    optimizer.apply_gradients(zip(reduced_grads, trainable_variables))
            
    return loss, out

def model_infer(esm_feat, label, my_model):
    out = my_model(esm_feat, label, training=False)
    return out
    
def main():
    # Initialize Horovod
    hvd.init()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print("rank: ", hvd.local_rank(), "binding ", gpus[hvd.local_rank()//4:hvd.local_rank()//4+1])
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()//4:hvd.local_rank()//4+1], 'GPU')

    params = {}
    params["batch_size"] = 1
    params["d_out"] = 538
    params["d_feat"] = 1280
    params["data_list"] = "/work/home/xugang/projects/weak/go/data/EC_split_seq_embedding_label.pkl"
    params["esm_dir"] = "/work/home/xugang/projects/weak/go/data/esm_feat_ec"
    params["save_path"] = r'./models'
    
    data_train_gen = DataGenerator(batch_size=1,
                                   data_set="train",
                                   params=params,
                                   rank=hvd.rank(),
                                   size=hvd.size())

    data_val_gen = DataGenerator(batch_size=1,
                                 data_set="val",
                                 params=params,
                                 rank=0,
                                 size=1)

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
    
    print (len(my_model.trainable_variables))
    if hvd.rank() == 0:
        my_model.save_model(name="0_weight.h5")   
    hvd.join()
    my_model.load_model(name="0_weight.h5")  
    
    learning_rate = 1.e-3
    lr = tf.Variable(tf.constant(learning_rate), name='lr', trainable=False)
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    epochs = 25
    best_epoch = 0
    best_f1 = 0
    count = 0
    for epoch in range(epochs):
        print('LR:', lr.numpy(), "count:", count)
        start_time = time.time()
        Precisions = [] 
        Recalls = []
        F1s = []
        for step, item in enumerate(data_train_gen):

            name, esm_feat, label = item

            loss, out = model_train(hvd, esm_feat, label, my_model, optimizer)
      
            Precision, Recall, F1 = calculate_metrics(label.numpy(), out)
            Precisions.append(Precision)
            Recalls.append(Recall)
            F1s.append(F1)

            run_time = time.time() - start_time
            start_time = time.time()
                
            if step % 100 == 0:
                print('Rank: %d, Epoch: %d, step: %d, loss: %3.3e, Precisions: %3.3e, Recalls: %3.3e, F1s: %3.3e, time: %3.3f'
                      % (hvd.rank(), epoch, step, loss.numpy(), np.mean(Precisions), np.mean(Recalls), np.mean(F1s), run_time)) 
                print (loss, Precision, Recall, F1)

        if hvd.rank() == 0:
            my_model.save_model(name=str(epoch)+"_weight.h5")   
        hvd.join()      
            
        start_time = time.time()
        Precisions = [] 
        Recalls = []
        F1s = []
        for step, item in enumerate(data_val_gen):

            name, esm_feat, label = item
            
            out = model_infer(esm_feat, None, my_model)
            
            Precision, Recall, F1 = calculate_metrics(label.numpy(), out)
            Precisions.append(Precision)
            Recalls.append(Recall)
            F1s.append(F1)

        print('Val, Rank: %d, Epoch: %d, Precisions: %3.3e, Recalls: %3.3e, F1s: %3.3e, time: %3.3f'
              % (hvd.rank(), epoch, np.mean(Precisions), np.mean(Recalls), np.mean(F1s), run_time)) 
        
        val_f1 = np.mean(F1s)
        if val_f1 > best_f1:
            best_epoch = epoch
            best_f1 = val_f1
            if hvd.rank() == 0:
                my_model.save_model(name="best_weight.h5")   
            hvd.join()     
            print ("best_epoch", best_epoch, "best_f1", best_f1)
        else:
            lr.assign(lr/2)
            count += 1
        
        start_time = time.time()
        Precisions = [] 
        Recalls = []
        F1s = []
        for step, item in enumerate(data_test_gen):

            name, esm_feat, label = item
            
            out = model_infer(esm_feat, None, my_model)
            
            Precision, Recall, F1 = calculate_metrics(label.numpy(), out)
            Precisions.append(Precision)
            Recalls.append(Recall)
            F1s.append(F1)

        print('Test, Rank: %d, Epoch: %d, Precisions: %3.3e, Recalls: %3.3e, F1s: %3.3e, time: %3.3f'
              % (hvd.rank(), epoch, np.mean(Precisions), np.mean(Recalls), np.mean(F1s), run_time)) 
        
        if count == 4:
            print ("best_epoch", best_epoch)
            break
            
if __name__ == '__main__':
    print('Running training through horovod.run')
    main()
    