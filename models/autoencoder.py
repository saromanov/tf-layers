import tensorflow as tf


class Autoencoder:
    """definition for encoder and decoder
    """

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def encoder(self, activation=tf.nn.sigmoid, activations=[], weights_names=[], biases_names=[]):
        """encoder though network model and returns result layer

        :param activation: set activation function which will be 
            applying to all layers
        :param activations: activations will be applying to each layer
            number of activations should be equal with number of layers
        """
        if len(activations) != 0 and len(activations) != len(weights_names):
            raise Exception('number of activation functions is not equal to number of layers')
    
        result = None
        for i in range(len(weights_names)):
            if len(activation) > 0:
                activation = activations[i]
            result = activation(tf.add(tf.matmul(x, self.weights[self.weights_names[i]]), self.biases[self.biases_names[i]]))
        return result
