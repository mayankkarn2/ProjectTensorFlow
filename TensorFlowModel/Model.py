
from tensorflow.python.keras import models , optimizers , losses ,activations
from tensorflow.python.keras.layers import *
import tensorflow as tf
import time
from tensorflow.keras.layers import LeakyReLU
from functools import partial, update_wrapper
import sys 


def wrapped_partial(func, *args, **kwargs):
	# the rectifier is an activation function defined as the positive part of its argument
	# Rectified linear units find applications in computer vision and speech recognition[8][9] using deep neural nets.
    partial_func = partial(tf.nn.leaky_relu, *args, **kwargs)
    update_wrapper(partial_func, tf.nn.leaky_relu)
    return partial_func

class Classifier (object) :

    def __init__( self , number_of_classes , maxlen ):

        tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )
        # leak_partial = partial(tf.nn.leaky_relu, alpha=0.2)
        dropout_rate = 0.5
        base = wrapped_partial(tf.nn.leaky_relu, alpha=0.2)
        # print(base.func)
        input_shape = ( maxlen , )
        target_shape = ( maxlen , 1 )
        self.model_scheme = [
            Reshape( input_shape=(input_shape) , target_shape=( maxlen , 1 ) ),
            Conv1D( 128, kernel_size=2 , strides=1, activation=base, kernel_regularizer='l1'),
            # to indentify important features in the samples
            MaxPooling1D(pool_size=2 ),
            # to convert ndarray to 1D array 
            Flatten() ,
            # Dense - A linear operation in which every input is connected to every output by a weight (so there are n_inputs * n_outputs weights - which can be a lot!). Generally followed by a non-linear activation function
            Dense( 64 , activation=base) ,
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense( number_of_classes, activation=tf.nn.softmax)
        ]
        # sequential is a sequence of layers
        self.__model = tf.keras.Sequential(self.model_scheme)
        self.__model.compile(optimizer=optimizers.Adam( lr=0.0001 ),loss=losses.categorical_crossentropy ,metrics=[ 'accuracy' ] ,)


    def fit(self, X, Y ,  hyperparameters  ):
        initial_time = time.time()
        self.__model.fit( X , Y ,
                         batch_size=hyperparameters[ 'batch_size' ] ,
                         epochs=hyperparameters[ 'epochs' ] ,
                         callbacks=hyperparameters[ 'callbacks'],
                         validation_data=hyperparameters[ 'val_data' ]
                         )
        final_time = time.time()
        eta = ( final_time - initial_time )
        time_unit = 'seconds'
        if eta >= 60 :
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary( )
        print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format( hyperparameters[ 'epochs' ] , eta , time_unit ) )
    
    def evaluate(self , test_X , test_Y  ) :
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X  ):
        predictions = self.__model.predict( X  )
        return predictions

    def save_model(self , file_path ):
        self.__model.save(file_path)


    def load_model(self , file_path ):
        self.__model = models.load_model(file_path)
