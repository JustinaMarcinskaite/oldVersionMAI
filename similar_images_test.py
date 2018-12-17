import numpy as np
import os
import tensorflow as tf

import multiprocessing as mp
import gym
import gym.spaces #TODO: remove this (only used to suppress warning

import scipy.ndimage
import matplotlib.pyplot as plt

import gym_utils
import data_utils


import data_utils
from exp_parameters import ExpParam
from train_vision import create_or_load_vae

print("Load test - Begin.")
file_name = 'Mnist_latent_train_0000'
data_path = './data/' + file_name + '.h5'
print(data_path)
data = data_utils.load_h5_as_list(data_path)
z, _, action, reward, done = data

### Settings
model_name = 'mnist_discrete_LAT64(2)_MADE1543521662'

latent = [[32*64, 2]]
raw_dim = (210, 160, 3)
net_dim = (32*4, 32*3, 3)


### Do stuff
## DISCRETE
exp_param = ExpParam(
    lat_type="discrete",
    dataset='mnist',
    latent=[[64 * 32, 2]],
    raw_type=tf.float32,
    raw_dim=raw_dim,
    net_dim=net_dim,
    learning_rate=0.001,
    # batch_size=16,
    max_example=1e5,
)

### Load model
model_path = 'saved_model/'
model_path += model_name

sess, network, _ = create_or_load_vae(model_path, exp_param=exp_param, critical_load=True)

### Encode data
print('\tEncoding...')
reconstruction = sess.run([network.z, network.latent_var], feed_dict={
    network.raw_input: obs,
    network.mask_in: obs_mask,
    network.is_training: False
})

#Show image before and after




 #another test is to prove that the ball and bricks are captured correctly.

