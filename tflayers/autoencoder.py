import tensorflow as tf


class Autoencoder:
    """definition for encoder and decoder
    """

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def encoder(self, x, activation=tf.nn.sigmoid, activations=[],
                weights_names=[], biases_names=[]):
        """encoder though network model and returns result layer

        :param x: input data which should be in tensorflow Graph format
        :param activation: set activation function which will be
            applying to all layers
        :param activations: activations will be applying to each layer
            number of activations should be equal with number of layers
        """
        if len(activations) != 0 and len(activations) != len(weights_names):
            raise Exception(
                'number of activation functions is not equal to number of layers')

        result = x
        for i in range(len(weights_names)):
            if len(activation) > 0:
                activation = activations[i]
            result = activation(tf.add(tf.matmul(result,
                                                 self.weights[self.weights_names[i]]),
                                       self.biases[self.biases_names[i]]))
        return result

class VariationalAutoencoder:

    def __init__(self, n_input, n_hidden, weights, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.weights = weights
    
    def make(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        return optimizer.minimize(self.cost)

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

