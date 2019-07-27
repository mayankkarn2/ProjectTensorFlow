import tensorflow as tf
# from keras.layers import LeakyReLU

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
keras_model_path =  'C:\\Users\\91998\\OneDrive\\Desktop\\TensorFlowActual\\models\\model.h5'
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
converter.post_training_quantize = True
tflite_buffer = converter.convert()
open( 'android/model.tflite' , 'wb' ).write( tflite_buffer )

print('TFLite model created.')