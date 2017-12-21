# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:47:31 2016

@author: rob

Code base from
https://github.com/rinuboney/ladder

- Why start backpropping from the softmaxed-layer?

"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import input_data
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
time_start_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

convmodes = ['convFC','convSmall']
convmode = convmodes[1]

logsave = True   # Do you want log files and checkpoint savers?
vis = False        #Visualize the Original - Noised - Recovered for the unsupervised samples
retore_ckpt = False


#map_sizes = [26,1,1,1,1,1,3]
''' origin
map_sizes = [28]
for c in range(1,C):
  map_sizes += [map_sizes[c-1]-2]
'''
num_examples = 60000
num_epochs = 150
num_labeled = 100
num_classes = 10

starter_learning_rate = 0.002

decay_after = int(num_epochs*0.67)  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs  # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, 784))
images = tf.reshape(inputs,[-1,28,28,1])
outputs = tf.placeholder(tf.float32)

#WZN: switch between convolution modes. NOTE: meanpool is supposed to be put at last or will cause error because of softmax
if convmode == convmodes[0]:
  channel_sizes = [1,1000,500,250,250,250,10,10]  #This is 'ConvFC'
  conv_kernels = [26,1,1,1,1,1,3]
  conv_types = ['convv','convv','convv','convv','convv','convv','meanpool']
  act_types =  ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'softmax']
  denoising_layers=['gauss','gauss','gauss','gauss','gauss','gauss','gauss','gauss']
  denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
elif convmode == convmodes[1]:
  channel_sizes = [1,32,32,64,64,64,128,10,10,10]  #This is 'ConvFC'
  conv_kernels = [5,2,3,3,2,3,1,6,1] #Note here the last fc is replaced by convv
  conv_types = ['convf','maxpool','convv','convf','maxpool','convv','convv','meanpool','convv']
  act_types =  ['relu', 'linear', 'relu', 'relu', 'linear', 'relu', 'relu', 'linear',  'softmax']
  pad_val = [None,2,None,None,1,None,None,None,None]
  denoising_layers=[None,None,None,None,None,None,None,None,None,'gauss']
  denoising_cost = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

C = len(channel_sizes) - 1 # number of channel sizes

def bi(inits, size, name):
  return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
  return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

def conv2d(inputs,output_dim,kernel_size,strides,padding='valid',
          activation=None,kernel_initializer = tf.uniform_unit_scaling_initializer(factor=1.43),#
          name=None,reuse=False):
  output = tf.layers.conv2d(inputs,output_dim,kernel_size,strides,padding=padding,
          activation=activation,kernel_initializer = kernel_initializer,name=name,reuse=reuse)
  return output

def deconv2d(inputs,output_dim,kernel_size,
          strides,activation=None,kernel_initializer = tf.uniform_unit_scaling_initializer(factor=1.43),#tf.contrib.layers.xavier_initializer_conv2d(seed=None),#
          name=None):
  output = tf.layers.conv2d_transpose(inputs,output_dim,kernel_size,strides,
          activation=activation,kernel_initializer = kernel_initializer,name=name)
  return output

weights = {'W': [0]*C,
           'V': [0]*C,
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, channel_sizes[l+1], "beta") for l in range(C)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, channel_sizes[l+1], "beta") for l in range(C)]}
''' origin, now use conv2d and xavier
initi = tf.uniform_unit_scaling_initializer(factor=1.43)
for c in range(C):
  if c == C-1: #Make the kernel as big as the final map sizes. Similar to Fully Convolutional Layer
    width = map_sizes[-1]
    shape = [width,width,channel_sizes[c],channel_sizes[c+1]]
    weights['W'][c] = tf.get_variable(name='W'+str(c), shape=shape, initializer = initi)
  else:
    shape = [3,3,channel_sizes[c],channel_sizes[c+1]]
    weights['W'][c] = tf.get_variable(name='W'+str(c), shape=shape, initializer = initi)
  print('W%s has shape '%c+str(shape))
for c in range(C-1,-1,-1):
  if c == C-1:
    width = map_sizes[-1]
    shape = [width,width,channel_sizes[c],channel_sizes[c+1]]
    weights['V'][c] = tf.get_variable(name='V'+str(c), shape=shape, initializer = initi)
  else:
    shape = [3,3,channel_sizes[c],channel_sizes[c+1]]
    weights['V'][c] = tf.get_variable(name='V'+str(c), shape=shape, initializer = initi)
  print('V%s has shape '%c+str(shape))
'''


noise_std = 0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer

#Note, these four functions work now for 4D Tensors
join = lambda l, u: tf.concat([l, u], 0)
labeled = lambda x: tf.slice(x, [0, 0,0,0], [batch_size, -1,-1,-1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0,0,0], [-1, -1,-1,-1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))
#The old functions that work with 2D Tensors
labeled2 = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled2 = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.9999)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None,axes=[0,1,2]):
  """Set axes to [0] for batch-norm in a 2D Tensor"""
  if mean is None or var is None:
      mean, var = tf.nn.moments(batch, axes=axes)
  return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[c]), trainable=False) for c in channel_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[c]), trainable=False) for c in channel_sizes[1:]]


def update_batch_normalization(batch, l,axes=[0,1,2]):
  "batch normalize + update average mean and variance of layer l"
  mean, var = tf.nn.moments(batch, axes=axes)
  assign_mean = running_mean[l-1].assign(mean)
  assign_var = running_var[l-1].assign(var)
  bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
  with tf.control_dependencies([assign_mean, assign_var]):
    return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(images, noise_std, reuse=True):
  h = images + tf.random_normal(tf.shape(images)) * noise_std  # add noise to input
  d = {}  # to store the pre-activation, activation, mean and variance for each layer
  # The data for labeled and unlabeled examples are stored separately
  d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
  for l in range(1, C+1): #WZN:note the last one is mean pool not conv2d
    print "Layer ", l, ": ", channel_sizes[l-1], " -> ", channel_sizes[l]
    d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
#    z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation

    kernel_size = [conv_kernels[l-1],conv_kernels[l-1]]
    if conv_types[l-1]=='meanpool':
      z_pre = tf.nn.pool(h, kernel_size, 'AVG', 'VALID', name='globalmeanpool')
    elif conv_types[l-1]=='maxpool':
      pad1 = pad_val[l-1]
      pad2 = pad_val[l-1]
      h = tf.pad(h,[[0,0],[pad1,pad1],[pad2,pad2],[0,0]])
      z_pre = tf.nn.pool(h, kernel_size, 'MAX', 'VALID', strides=kernel_size, name='maxpooling'+str(l))
    elif conv_types[l-1]=='convv': #assume others are convf or convv
      z_pre = conv2d(h, channel_sizes[l], kernel_size ,[1, 1], name='ecoder_'+str(l-1),reuse=reuse)
    elif conv_types[l-1]=='convf': #assume others are convf or convv
      z_pre = conv2d(h, channel_sizes[l], kernel_size ,[1, 1], padding='same', name='ecoder_'+str(l-1),reuse=reuse)  
    print z_pre.shape
    z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

    m, v = tf.nn.moments(z_pre_u, axes=[0,1,2]) #in size [,channel_sizes[l]]

    # if training:
    def training_batch_norm():
      # Training batch normalization
      # batch normalization for labeled and unlabeled examples is performed separately
      if noise_std > 0:
        # Corrupted encoder
        # batch normalization + noise
        z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
        z += tf.random_normal(tf.shape(z_pre)) * noise_std
      else:
        # Clean encoder
        # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
        z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
      return z

    # else:
    def eval_batch_norm():
      # Evaluation batch normalization
      # obtain average mean and variance and use it to normalize the batch
      mean = ewma.average(running_mean[l-1])
      var = ewma.average(running_var[l-1])
      z = batch_normalization(z_pre, mean, var)
      # Instead of the above statement, the use of the following 2 statements containing a typo
      # consistently produces a 0.2% higher accuracy for unclear reasons.
      # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
      # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
      return z

    # perform batch normalization according to value of boolean "training" placeholder:
    z = tf.cond(training, training_batch_norm, eval_batch_norm)
    if act_types[l-1]=='softmax':
      # use softmax activation in output layer
      z = z * np.float32(3) * np.float32(np.sqrt(3)) # ind the theano version
      h = tf.nn.softmax(weights['gamma'][l-1] * (tf.squeeze(z,axis=[1,2]) + weights["beta"][l-1]))

      h_logit = tf.squeeze(labeled(weights['gamma'][l-1] * (z + weights["beta"][l-1])),axis=[1,2])
      h = tf.expand_dims(tf.expand_dims(h,axis=1),axis=2)
    elif act_types[l-1]=='relu':
      # use ReLU activation in hidden layers
      h = tf.nn.relu(z + weights["beta"][l-1])
    elif act_types[l-1]=='linear':
      h = z
    d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
    d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
  d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
  return h, d, h_logit

print "=== Corrupted Encoder ==="
y_c, corrupted, y_c_logit = encoder(images, noise_std,reuse=False)

print "=== Clean Encoder ==="
y, clean, y_logit = encoder(images, 0.0, reuse=True)  # 0.0 -> do not add noise

print "=== Decoder ==="


def g_gauss(z_c, u, size):
  """gaussian denoising function proposed in the original paper
  z_c: corrupted latent variable
  u: Tensor from layer (l+1) in decorder
  size: number hidden neurons for this layer"""
  wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
  a1 = wi(0., 'a1')
  a2 = wi(1., 'a2')
  a3 = wi(0., 'a3')
  a4 = wi(0., 'a4')
  a5 = wi(0., 'a5')

  a6 = wi(0., 'a6')
  a7 = wi(1., 'a7')
  a8 = wi(0., 'a8')
  a9 = wi(0., 'a9')
  a10 = wi(0., 'a10')
  #Crazy transformation of the prior (mu) and convex-combi weight (v)
  mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5  #prior
  v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10  #convex-combi weight

  z_est = (z_c - mu) * v + mu  #equation [2] in http://arxiv.org/pdf/1507.02672v2.pdf
  return z_est

# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers
d_costs_0 = []
#shape_z = []
for l in range(C, -1, -1):
  print "Layer ", l, ": ", 'denoise_mode:', denoising_layers[l], channel_sizes[l+1] if l+1 < len(channel_sizes) else None, " -> ", channel_sizes[l], ", denoising cost: ", denoising_cost[l]
  z, z_c = clean['unlabeled']['z'][l], corrupted['unlabeled']['z'][l]
  m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
  # m are batch-norm means, v are batch-norm stddevs
  if l == C:
    u = unlabeled(y_c)
  else:
    #WZN: deavgpooling is deconv
    #print l
    if denoising_layers[l]=='gauss':
      kernel_size = [conv_kernels[l],conv_kernels[l]]
      u = deconv2d(z_est[l+1], channel_sizes[l], kernel_size ,[1, 1], name='decoder'+str(l))
    else:
      kernel_size=None
      u = None
    #(z_est[l+1], weights['V'][l], tf.stack([tf.shape(z_est[l+1])[0], map_sizes[l], map_sizes[l], channel_sizes[l]]),strides=[1, 1, 1, 1], padding='VALID',name = 'CT'+str(l))
  if denoising_layers[l]=='gauss':
    u = batch_normalization(u)
    z_est[l] = g_gauss(z_c, u, channel_sizes[l])
    z_est_bn = (z_est[l] - m) / tf.sqrt(v+1e-10)
    # append the cost of this layer to d_cost
    #shape_z.append([tf.constant(channel_sizes[l]),tf.reduce_prod(tf.cast(tf.shape(z)[1:],tf.float32))])
    d_cost_norm = tf.reduce_mean(tf.square(z_est_bn - z)) 
    d_cost.append(d_cost_norm * denoising_cost[l])
    d_costs_0.append(d_cost_norm)
  else:
    z_est[l] = None
    z_est_bn = None

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)

#Convert y* back to 2D Tensor
y_N = tf.squeeze(y_N, squeeze_dims=[1,2])
y = tf.squeeze(y, squeeze_dims=[1,2])
s_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs,logits=y_c_logit))
#s_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
loss = s_cost + u_cost  # total cost

#pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction
correct_prediction = tf.equal(tf.argmax(labeled2(y), 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

correct_prediction_val = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print "===  Loading Data ==="
mnist = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True, val_ratio = 0.15)

if logsave: saver = tf.train.Saver()

print "===  Starting Session ==="
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path and logsave and retore_ckpt:
  # if checkpoint exists, restore the parameters and set epoch_n and i_iter
  saver.restore(sess, ckpt.model_checkpoint_path)
  epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
  i_iter = (epoch_n+1) * (num_examples/batch_size)
  print "Restored Epoch ", epoch_n
else:
  # no checkpoint exists. create checkpoints directory if it does not exist.
  if not os.path.exists('checkpoints'):
      os.makedirs('checkpoints')
  init = tf.global_variables_initializer()
  sess.run(init)

print "=== Training ==="
#print "Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"
acc_ma = 0.0
s_cost_ma = 0.0
u_cost_ma = 0.0
images, labels = mnist.train.next_batch(batch_size)
#debug = sess.run([clean,z_c,z_est_bn,z_est,d_costs_0], feed_dict={inputs: images, outputs: labels, training: True})

for i in range(i_iter, num_iter):
  images, labels = mnist.train.next_batch(batch_size)
  result = sess.run([train_step,accuracy,s_cost,u_cost], feed_dict={inputs: images, outputs: labels, training: True})
  acc_ma = 0.8*acc_ma+0.2*result[1]
  s_cost_ma = 0.8*s_cost_ma+0.2*result[2]
  u_cost_ma = 0.8*u_cost_ma+0.2*result[3]

#  print(debug)
  if  i%100 == 0:  #((i+1) % (num_iter/num_epochs) == 0)
    epoch_n = i/(num_examples/batch_size)
    if (epoch_n+1) >= decay_after:
      # decay learning rate
      # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
      ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
      ratio = max(0, ratio / (num_epochs - decay_after))
      sess.run(learning_rate.assign(starter_learning_rate * ratio))
    if logsave: saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
    fetch = [accuracy,s_cost,u_cost,d_costs_0,y_c_logit,y,outputs]
    #if vis: fetch += [corrupted['unlabeled']['z'][0],z_est[0]]
    images_val, labels_val = mnist.validation.next_batch(batch_size)
    result = sess.run(fetch, feed_dict={inputs: images_val, outputs:labels_val, training: False})
    #import pdb; pdb.set_trace()
    print(("At %5.0f of %5.0f acc %.3f cost super %5.3f unsuper %5.3f den_cost "+','.join('%5.3f' % v for v in list(reversed(result[3]))))%(i,num_iter,result[0],result[1],result[2]))
    if result[0]<20 and i>500:
      import pdb; pdb.set_trace()
      #print result[-1]
      #print result[-2]
    #Visualize
    if vis and i%100 == 0:
      Nplot = 3
      ind = np.random.choice(num_labeled,Nplot)
      f, axarr = plt.subplots(Nplot, 3)
      for r in range(Nplot):
        axarr[r, 0].imshow(np.reshape(images_val[batch_size+ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 1].imshow(np.reshape(result[3][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 2].imshow(np.reshape(result[4][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        plt.setp([a.get_xticklabels() for a in axarr[r, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[r, :]], visible=False)
      f.subplots_adjust(wspace=0.0, hspace = 0.0)
      f.suptitle('Original - Corrupted - Recovered')
      plt.show()


    if logsave and i%200==0:
      with open('train_log_conv_'+str(num_labeled)+'_'+time_start_str, 'ab') as train_log:
        # write test accuracy to file "train_log"
        train_log_w = csv.writer(train_log)
        log_i = [epoch_n] + sess.run([accuracy_val], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
        train_log_w.writerow(log_i)

print "Final Accuracy: ", sess.run(accuracy_val, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

sess.close()
