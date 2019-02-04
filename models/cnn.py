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
    
    def make(self):
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        return tf.layers.dense(fc1, n_classes)