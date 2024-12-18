# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:56:34 2023

@author: whuxu
"""
import numpy as np
import tensorflow as tf
import os

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, d_ffn):
        super(FeedForward, self).__init__()

        self.fc1 = tf.keras.layers.Dense(d_ffn, use_bias=False)
        self.fc2 = tf.keras.layers.Dense(dim, use_bias=False)
        self.fc3 = tf.keras.layers.Dense(d_ffn, use_bias=False)
    
    def call(self, x):
        return self.fc2(tf.nn.swish(self.fc1(x)) * self.fc3(x))
    
class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):

        self.weight = self.add_variable(name="RMSNorm_w",
                                        shape=(self.dim,),
                                        initializer=tf.ones_initializer(),
                                        trainable=True)
        self.built = True

    def call(self, x):
        output = x * tf.math.rsqrt(tf.reduce_mean(tf.math.square(x), -1, keepdims=True) + self.eps)
        return output * self.weight
    
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def loss_function(label, logit, L, d_out):
    # label (1,params["d_out"])
    # logit (1, L, params["d_out"])
        
    n_top = min(int(L*0.1), 10)
    n_top = max(n_top, 1)
    n_label = d_out
    n_len = L
    
    y_true = label
    y_pred = logit
    y_pred_sigmoid = tf.sigmoid(y_pred)
    
    y_pred_sigmoid2 = tf.reshape(tf.transpose(y_pred_sigmoid, (0,2,1)), -1)
    
    indice_w_label = tf.where(y_true==1)[:,1]
    
    indice_1s = np.zeros(n_len*n_label)
    for col in indice_w_label.numpy():
        indice_1 = np.argsort(y_pred_sigmoid[0,:,col]).tolist()
        indice_1.reverse()
        indice_1 = np.array(indice_1[:n_top]) + col*n_len
        indice_1s[indice_1] = 1
    
    n_w_label = indice_w_label.shape[0]*n_top
    n_wo_label = min(n_len*n_label - n_w_label, n_w_label)
    
    y_pred_w_label = y_pred_sigmoid2[indice_1s==1]
    
    indice_wo_label = tf.where(indice_1s==0)[:,0]
    indice_wo_label = tf.random.shuffle(indice_wo_label)[:n_wo_label]
    
    y_pred_wo_label = tf.gather(y_pred_sigmoid2, indice_wo_label)
    
    y_true2 = tf.constant([1]*n_w_label + [0]*n_wo_label, dtype=tf.float32)
    y_pred2 = tf.concat([y_pred_w_label, y_pred_wo_label], 0)
    
    loss = loss_fn(y_true2, y_pred2)
    return loss

def loss_function_global(label, logit, L, d_out, col_select=20):
    # label (1,params["d_out"])
    # logit (1, L, params["d_out"])
        
    n_top = min(int(L*0.1), 10)
    n_top = max(n_top, 1)
    n_label = d_out
    n_len = L
    
    y_true = label
    y_pred = logit
    y_pred_sigmoid = tf.sigmoid(y_pred)
    
    if n_label > 5000:
        y_pred_sigmoid_max = np.max(y_pred_sigmoid.numpy(), 1)
        y_pred_max_n = np.sort(y_pred_sigmoid_max[0])[-col_select]
        y_pred_max = tf.cast((y_pred_sigmoid_max > y_pred_max_n), tf.int32)
        y_pred_sigmoid2 = tf.reshape(tf.transpose(y_pred_sigmoid, (0,2,1)), -1)
        indice_wo_label = (y_true==0) & (y_pred_max==1)
        indice_wo_label = tf.where(indice_wo_label==True)[:,1]
    else:
        y_pred_sigmoid2 = tf.reshape(tf.transpose(y_pred_sigmoid, (0,2,1)), -1)
        indice_wo_label = tf.where(y_true==0)[:,1]
        
    indice_0s = np.zeros(n_len*n_label)
    for col in indice_wo_label.numpy():
        indice_0 = np.argsort(y_pred_sigmoid[0,:,col]).tolist()
        indice_0.reverse()
        indice_0 = np.array(indice_0[:n_top]) + col*n_len
        indice_0s[indice_0] = 1
    
    n_wo_label = indice_wo_label.shape[0]*n_top
    y_pred_wo_label = y_pred_sigmoid2[indice_0s==1]
    
    y_true2 = tf.constant([0]*n_wo_label, dtype=tf.float32)
    y_pred2 = y_pred_wo_label
    
    loss = loss_fn(y_true2, y_pred2)
    return loss

def calculate_metrics(y_true, y_pred):  
    
    y_pred = np.max(y_pred, 1)
            
    assert y_true.shape == y_pred.shape
    
    y_true = (y_true >= 0.5).astype(int)  
    y_pred = (y_pred >= 0.5).astype(int)  
  
    TP = np.sum((y_true == 1) & (y_pred == 1))  
    FP = np.sum((y_true == 0) & (y_pred == 1))  
    FN = np.sum((y_true == 1) & (y_pred == 0))  
    TN = np.sum((y_true == 0) & (y_pred == 0))  
  
    epsilon = 1e-7
    Precision = TP / (TP + FP + epsilon)  
    Recall = TP / (TP + FN + epsilon)  
    F1 = 2 * (Precision * Recall) / (Precision + Recall + epsilon)  
  
    return Precision, Recall, F1

class OPUSGO(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ffn,
                 d_out, drop_rate=0.5):
        
        super(OPUSGO, self).__init__()
        
        self.d_model = d_model
        self.d_out = d_out
        
        self.ffn_norm = RMSNorm(dim=d_model)
        self.ffn = FeedForward(d_model, d_ffn)     
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.final_layer = tf.keras.layers.Dense(d_out)

    def call(self, inputs, label, training=False):
        # inputs (1, L, params["d_feat"])
        # label (1, params["d_out"])
        
        L = inputs.shape[1]
        assert inputs.shape == (1, L, self.d_model)
        
        decode_out = self.ffn(inputs)
        decode_out = tf.nn.swish(decode_out)
        decode_out = self.ffn_norm(decode_out)
        decode_out = self.dropout(decode_out, training=training)
        logit = self.final_layer(decode_out) # (1, L, params["d_out"])
        
        out = tf.sigmoid(logit)
        out = out.numpy()
        
        if training:
            assert label.shape == (1, self.d_out)
            loss = loss_function(label, logit, L, self.d_out) + loss_function_global(label, logit, L, self.d_out)
            return loss, out
        else:
            return out

    def load_model(self, name=None):
        print ("load_weights", name)
        self.load_weights(os.path.join('./models/', name))
                
    def save_model(self, name=None):
        print ("save_weights", name)
        self.save_weights(os.path.join('./models/', name),
                          save_format="h5")
