import tensorflow as tf
#from tensorflow.python import control_flow_ops
import input_data
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
time_start_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

logsave = True   # Do you want log files and checkpoint savers?
vis = False        #Visualize the Original - Noised - Recovered for the unsupervised samples
retore_ckpt = False

L = len(layer_sizes) - 1  # number of layers

num_examples = 60000
num_epochs = 200
num_labeled = 1000#100

#origin: 0.02
starter_learning_rate = 0.002

#WZN: origin 15
decay_after = int(num_epochs*0.67)  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs  # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32,  shape=(None, layer_sizes[-1]))


def bi(inits, size, name):
  return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
  return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers

weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
           'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "gamma") for l in range(L)]}

noise_std = 0.2#0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer
denoising_cost = [2000.0, 20.0, 0.10, 0.10, 0.10, 0.10, 0.10]#[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: tf.concat([l, u], 0)
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)
#origin 0.99
ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None):
  if mean is None or var is None:
      mean, var = tf.nn.moments(batch, axes=[0])
  return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]


def update_batch_normalization(batch, l):
  "batch normalize + update average mean and variance of layer l"
  mean, var = tf.nn.moments(batch, axes=[0])
  assign_mean = running_mean[l-1].assign(mean)
  assign_var = running_var[l-1].assign(var)
  bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
  with tf.control_dependencies([assign_mean, assign_var]):
    return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(inputs, noise_std):
  h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
  d = {}  # to store the pre-activation, activation, mean and variance for each layer
  # The data for labeled and unlabeled examples are stored separately
  d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
  d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
  for l in range(1, L+1):
    print "Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l]
    d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
    z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
    z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

    m, v = tf.nn.moments(z_pre_u, axes=[0])

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

    if l == L:
      # use softmax activation in output layer
      h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
      h_logit = labeled(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
    else:
      # use ReLU activation in hidden layers
      h = tf.nn.relu(z + weights["beta"][l-1])
    d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
    d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
  d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
  return h, d, h_logit

print "=== Clean Encoder ==="
y, clean, _ = encoder(inputs, 0.0)  # 0.0 -> do not add noise

print "=== Corrupted Encoder ==="
y_c, corrupted, y_c_logit = encoder(inputs, noise_std)


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
for l in range(L, -1, -1):
  print "Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l]
  z, z_c = clean['unlabeled']['z'][l], corrupted['unlabeled']['z'][l]
  m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
  # m are batch-norm means, v are batch-norm stddevs
  if l == L:
    u = unlabeled(y_c)
  else:
    u = tf.matmul(z_est[l+1], weights['V'][l])
  u = batch_normalization(u)
  z_est[l] = g_gauss(z_c, u, layer_sizes[l])
  z_est_bn = (z_est[l] - m) / tf.sqrt(v) #WZN: origin is v
  # append the cost of this layer to d_cost
  d_cost_norm = (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l])
  d_cost.append(d_cost_norm * denoising_cost[l])
  d_costs_0.append(d_cost_norm)

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
s_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs,logits=y_c_logit))#-tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost #origin: reduce_mean
loss = s_cost + u_cost  # total cost

#pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction
size1 = tf.shape(y)
correct_prediction = tf.equal(tf.argmax(labeled(y), 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

correct_prediction_val = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)


# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
#origin
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#with tf.control_dependencies([train_step]):
#    train_step = tf.group(bn_updates)
with tf.control_dependencies(bn_assigns):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
print "Initial Accuracy: ", sess.run(accuracy_val, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

for i in range(i_iter, num_iter):
  images, labels = mnist.train.next_batch(batch_size)
  debug = sess.run([size1,train_step], feed_dict={inputs: images, outputs: labels, training: True})
  #print(debug)
  #print images.shape, labels.shape
  if i%500 == 0:  #((i+1) % (num_iter/num_epochs) == 0)
    epoch_n = i/(num_examples/batch_size)
    if (epoch_n+1) >= decay_after:
      # decay learning rate
      # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
      ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
      ratio = max(0, ratio / (num_epochs - decay_after))
      sess.run(learning_rate.assign(starter_learning_rate * ratio))
    if logsave: saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
    fetch = [accuracy,s_cost,u_cost,d_costs_0]
    if vis: fetch += [corrupted['unlabeled']['z'][0],z_est[0]]
    images_val, labels_val = mnist.validation.next_batch(batch_size)
    #print labels_val
    result = sess.run(fetch, feed_dict={inputs: images_val, outputs:labels_val, training: False})
    print result[1]
    print(("At %5.0f of %5.0f acc %.3f cost super %5.3f unsuper %5.3f den_cost "+','.join('%5.3f' % v for v in list(reversed(result[3]))))%(i,num_iter,result[0],result[1],result[2]))
    #Visualize
    if vis:
      Nplot = 3
      ind = np.random.choice(num_labeled,Nplot)
      f, axarr = plt.subplots(Nplot, 3)
      for r in range(Nplot):
        axarr[r, 0].imshow(np.reshape(images_val[batch_size+ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 1].imshow(np.reshape(result[3][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)
        axarr[r, 2].imshow(np.reshape(result[4][ind[r]],(28,28)), cmap=plt.get_cmap('gray'),vmin = 0.0, vmax=1.0)

        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        plt.setp([a.get_xticklabels() for a in axarr[r, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[r, :]], visible=False)
      f.subplots_adjust(wspace=0.0, hspace = 0.0)
      f.suptitle('Original - Corrupted - Recovered')
      plt.show()


    if logsave and i%200==0:
      with open('train_log_'+str(num_labeled)+'_'+time_start_str, 'ab') as train_log:
        # write test accuracy to file "train_log"
        train_log_w = csv.writer(train_log)
        acc_out, den_costs0 = sess.run([accuracy_val,d_costs_0], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
        den_costs0 = list(reversed(den_costs0))
        #print (','.join('%5.3f' % v for v in den_costs0))
        log_i = [epoch_n]
        log_i.append(acc_out)
        log_i.append('denoise_loss: ')
        for v in den_costs0:
          log_i.append('%5.3f'%v)
        train_log_w.writerow(log_i)

print "Final Accuracy: ", sess.run(accuracy_val, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%"

sess.close()