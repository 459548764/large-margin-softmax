import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import Net
import struct
BATCH_SIZE=100
max_iter = 200
margin = 2

def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        if kind=='train':
                labels_path=os.path.abspath('/home/ubuntu/mnist/train-labels-idx1-ubyte')
                images_path=os.path.abspath('/home/ubuntu/mnist/train-images-idx3-ubyte')
        else:
                labels_path=os.path.abspath('/home/ubuntu/mnist/t10k-labels-idx1-ubyte')
                images_path=os.path.abspath('/home/ubuntu/mnist/t10k-images-idx3-ubyte')

        with open(labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II',
                                                                 lbpath.read(8))
                labels = np.fromfile(lbpath,
                                                         dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack(">IIII",
                                                                                           imgpath.read(16))
                images = np.fromfile(imgpath,
                                                         dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

X_train, y_train = load_mnist('../../mnist', kind='train')
mms=MinMaxScaler()
X_train=mms.fit_transform(X_train)
print X_train.shape
X_train=np.reshape(X_train,[60000,28,28,1])

batch_len =int( X_train.shape[0]/BATCH_SIZE)
batch_idx=0
train_idx=np.random.permutation(batch_len)


net = Net.Net(BATCH_SIZE, margin)
loss = net.loss()
train_step = tf.train.AdamOptimizer(0.004).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(max_iter):
	batch_shuffle_idx=train_idx[batch_idx]
	batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
	batch_ys=y_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
	
	if batch_idx<batch_len:
		batch_idx+=1
		if batch_idx==batch_len:
			batch_idx=0
	else:
		batch_idx=0
        _,loss_value=sess.run([train_step,loss], feed_dict={net.test_input:batch_xs, net.test_label: batch_ys})
	print loss_value
        if i % 10 == 0:
            [l] = sess.run(loss, feed_dict={net.test_input:batch_xs, net.test_label:batch_ys})
            print 'iter:', i, 'loss:', l

saver = tf.train.Saver()
saver.save(sess, './out/aaa')
sess.close()
