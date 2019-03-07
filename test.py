import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import Net
import struct

batch_size = 64
max_iter = 800
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
X_train=X_train[:3000]
y_train=y_train[:3000]
X_train=np.reshape(X_train,[3000,28,28,1])

def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  
	return onehot

net = Net.Net(batch_size, margin)
loss = net.network(tf.cast(X_train,tf.float32))
y_train=encode_labels(y_train,10)
pred_max=tf.argmax(loss,1)
y_max=tf.argmax(y_train,1)
correct_pred = tf.equal(pred_max,y_max)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,'./out/aaa')
	acc= sess.run(accuracy, feed_dict={net.test_input:X_train})
	print 'acc', acc
