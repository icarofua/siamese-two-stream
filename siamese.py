from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from utils import *
from sys import argv
#------------------------------------------------------------------------------
def siamese_model(type):
  if type=='plate':
    input_shape = (image_size_h_p,image_size_w_p,nchannels)
  else:
    input_shape = (image_size_h_c,image_size_w_c,nchannels)
  left_input = Input(input_shape)
  right_input = Input(input_shape)
  convnet = small_vgg(input_shape)
  encoded_l = convnet(left_input)
  encoded_r = convnet(right_input)

  # Add the distance function to the network
  L1_distance = L1_layer([encoded_l, encoded_r])

  prediction = Dense(2,activation='softmax')(L1_distance)
  optimizer = Adam(0.001, decay=2.5e-4)

  model = Model(inputs=[left_input,right_input],outputs=prediction)
  model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

  return model
#------------------------------------------------------------------------------
if __name__ == '__main__':
  run(siamese_model(argv[1]), argv[1])
