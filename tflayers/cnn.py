import tensorflow as tf


class CNN:
    """implementation of Convolutional Layer
    """
    def __init__(self, data, shape):
        self.layers = layer
        self.data = tf.reshape(x, data, shape)
    
    def add_layer(self, layer_type, xdim, ydim, activation=tf.nn.relu):
        if layer_type == 'conv':
            self.data = self.layers.append(tf.layers.conv2d(self.data, xdim, ydim, activation=activation))
        if layer_type == 'pool':
            self.data = self.layer.append(tf.layers.max_pooling2d(self.data, xdim, ydim))
    
    def make(self, dense, n_classes):
        fc1 = tf.contrib.layers.flatten(self.data)
        fc1 = tf.layers.dense(fc1, dense)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=True)
        return tf.layers.dense(fc1, n_classes)