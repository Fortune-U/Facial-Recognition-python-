# Custom L1 Distance layer module
# WHY DO WE NEED THIS : to load custom module

import tensorflow as tf 
from keras.layers import Layer

class L1Dist(Layer):
    
    #Init method inheritance 
    def __init__(self, **kwargs):
        super().__init__()
        
    # Magic happens here     
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)