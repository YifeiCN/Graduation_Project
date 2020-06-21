import tensorflow as tf
from tensorflow.keras import backend as K

class MaxoutConv2D(tf.keras.layers.Layer):
    
    
    def __init__(self, **kwargs):
        
        super(MaxoutConv2D, self).__init__(**kwargs)
        
        
    def call(self, x):
        
        output = K.max(x, axis=-1, keepdims=True)
        
        return output
    
    
    def compute_output_shape(self, input_shape):
        
        input_height = input_shape[1]
        
        input_width = input_shape[2]
        
        output_height = input_height
        
        output_width = input_width
        
        return (input_shape[0], output_height, output_width, 1)
    
    
    
# metric r2
def r2(y_true, y_pred):
    
    return 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))




