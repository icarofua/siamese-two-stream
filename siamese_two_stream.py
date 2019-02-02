from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Activation
from keras.optimizers import Adam
from utils import *

#------------------------------------------------------------------------------
def siamese_model(convnet_plate, convnet_car):
  left_input_P = Input((image_size_h_p,image_size_w_p,nchannels))
  right_input_P = Input((image_size_h_p,image_size_w_p,nchannels))
  left_input_C = Input((image_size_h_c,image_size_w_c,nchannels))
  right_input_C = Input((image_size_h_c,image_size_w_c,nchannels))
  encoded_l_P = convnet_plate(left_input_P)
  encoded_r_P = convnet_plate(right_input_P)
  encoded_l_C = convnet_car(left_input_C)
  encoded_r_C = convnet_car(right_input_C)

  # Add the distance function to the network
  L1_distanceP = L1_layer([encoded_l_P, encoded_r_P])
  L1_distanceC = L1_layer([encoded_l_C, encoded_r_C])
  concatL1 = Concatenate()([L1_distanceP, L1_distanceC])
  x = Dense(1024)(concatL1)
  x = Dropout(0.2)(x)
  x = Dense(512)(x)
  x = Dropout(0.2)(x)
  x = Dense(256)(x)
  x = Dropout(0.2)(x)
  x = Activation('relu')(x)
  predictionF2 = Dense(2,activation='softmax', name="fusion2_output")(x)
  optimizer = Adam(0.001, decay=2.5e-4)

  model = Model(inputs=[left_input_P, right_input_P, left_input_C, right_input_C], outputs=predictionF2)
  model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':
  run(siamese_model, None)
