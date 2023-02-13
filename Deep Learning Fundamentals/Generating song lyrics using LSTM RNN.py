import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()

with open("ZaynLyrics.txt", "r") as f:
  data = f.read()
  data = data.replace('\n', '')
  data = data.lower()

all_chars = list(set(data))
unique_chars = len(all_chars)
total_chars = len(data)
char_to_ix = {ch: i for i, ch in enumerate(all_chars)}
ix_to_char = {i: ch for i, ch in enumerate(all_chars)}

def generate_batch(seq_length, i):
  inputs = [char_to_ix[ch] for ch in data[i: i + seq_length]]
  targets = [char_to_ix[ch] for ch in data[i+1: i+seq_length+1]]
  inputs = np.array(inputs).reshape(seq_length, 1)
  targets = np.array(targets).reshape(seq_length, 1)
  return inputs, targets

seq_length = 25
learning_rate = 0.1
num_nodes = 300

def build_run(x):
  cell = tf.compat.v1.keras.layers.LSTMCell(units=num_nodes, activation=tf.compat.v1.nn.relu)
  outputs, states = tf.compat.v1.nn.dynamic_rnn(cell, x, dtype=tf.compat.v1.float32)
  return outputs, states

X = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1])
Y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1])

X = tf.compat.v1.cast(X, tf.compat.v1.int32)
Y = tf.compat.v1.cast(Y, tf.compat.v1.int32)

X_onehot = tf.compat.v1.one_hot(X, unique_chars)
Y_onehot = tf.compat.v1.one_hot(Y, unique_chars)

outputs, states = build_run(X_onehot)
outputs = tf.compat.v1.transpose(outputs, perm=[1, 0, 2])
W = tf.compat.v1.Variable(tf.compat.v1.random_normal((num_nodes, unique_chars), stddev=0.001))
B = tf.compat.v1.Variable(tf.compat.v1.zeros((1, unique_chars)))
Ys = tf.compat.v1.matmul(outputs[0], W) + B
prediction = tf.compat.v1.nn.softmax(Ys)
cross_entropy = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=Ys))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

def predict(seed, i):
  x = np.zeros((1, 1))
  x[0][0] = seed
  indices = []
  for t in range(i):
    p = sess.run(prediction, {X:x})
    index = np.random.choice(range(unique_chars), p=p.ravel())
    x[0][0] = index
    indices.append(index)
  return indices

batch_size = 100
total_batch = int(total_chars//batch_size)
epochs = 1000
shift = 0

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
  sess.run(init)
  for epoch in range(epochs):
    print("Epoch {}:".format(epoch))
    if shift + batch_size + 1 >= len(data):
      shift = 0
    
    # Get the input and target for each batch by generate_batch function which shifts the input 
    # by shift value and form target
    for i in range(total_batch):
      inputs, targets = generate_batch(batch_size, shift)
      shift += batch_size
      # Calculate loss
      if (i % 100 == 0):
        loss = sess.run(cross_entropy, feed_dict={X:inputs, Y:targets})

        # We get index of next predicted character by the predict function
        index = predict(inputs[0], 200)

        # pass the index to our ix_to_char dictionary and get the char
        txt = ''.join(ix_to_char[ix] for ix in index)
        print('Iteration %i: '%(i))
        print('\n %s \n' % (txt, ))
      
      sess.run(optimizer, feed_dict={X:inputs, Y:targets})