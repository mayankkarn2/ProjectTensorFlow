
from tensorflow.python.keras.callbacks import TensorBoard
from Model import Classifier
import numpy as np
import time
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


X = np.load( 'processed_data/x.npy')
Y = np.load( 'processed_data/y.npy')
test_X = np.load( 'processed_data/test_x.npy')
test_Y = np.load( 'processed_data/test_y.npy')

print( X.shape )
print( Y.shape )
print( test_X.shape )
print( test_Y.shape )

classifier = Classifier( number_of_classes=2 , maxlen=35 )
# classifier.load_model( 'models/model.h5' )

parameters = {
    'batch_size' : 100 ,
    'epochs' : 100 ,
    'callbacks' : [ TensorBoard( log_dir=os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)) ] ,
    'val_data' : ( test_X , test_Y )
}
classifier.fit( X , Y , parameters )
classifier.save_model( 'models/model.h5')

loss , accuracy = classifier.evaluate( test_X , test_Y )
print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )

