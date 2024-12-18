import numpy as np
import tensorflow as tf
import pickle
import os

class DataGenerator(tf.data.Dataset):
    def _generator(esm_dir:str, data_list:str, 
                   d_feat:int, d_out:int, 
                   data_set:str, rank:int, size:int):
        
        esm_feat_path = esm_dir.decode('utf-8')
        
        data_list = data_list.decode('utf-8')
        with open(data_list, 'rb') as file:
            dataset = pickle.load(file)

        data_set = data_set.decode('utf-8')
        dataset = dataset[data_set]
        print ("file length: ", len(dataset))

        num_files = len(dataset)//size
        dataset = dataset[rank*num_files:(rank+1)*num_files]
        print("keeping ", rank*num_files, (rank+1)*num_files, "for ", rank)
        
        if data_set == 'train':
            print ("data shuffle...")
            np.random.shuffle(dataset)

        for sample_idx in range(len(dataset)):
            
            name = dataset[sample_idx]['name']
            label = np.array(dataset[sample_idx]['label'])
            seq = dataset[sample_idx]['seq']
            
            esm_feat = np.load(os.path.join(esm_feat_path, name+".esm.npz"))['l']
            assert esm_feat.shape == (len(seq), d_feat)    
            assert label.shape == (d_out,)    
            
            yield name, esm_feat, label               
    
    def __new__(cls, batch_size:int = 32, data_set:str = '', params:dict = {}, 
                rank:int = 0, size:int = 1):

        ds = tf.data.Dataset.range(1)
        ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(
                                     cls._generator,
                                     output_types = (tf.string, tf.float32, tf.float32),
                                     output_shapes = ((),
                                                     (None, params["d_feat"]), 
                                                      (params["d_out"],)),
                                     args = (params["esm_dir"], params["data_list"], 
                                             params["d_feat"], params["d_out"], 
                                             data_set, rank, size)),
                           num_parallel_calls = 2)
        return ds.batch(batch_size).prefetch(batch_size*4)               
    