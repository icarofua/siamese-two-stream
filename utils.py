from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from config import *
from math import ceil
import json
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from collections import Counter
from keras import backend as K
from keras.layers import *
from keras.models import Model

#------------------------------------------------------------------------------
# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

#------------------------------------------------------------------------------
def tfs(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input1)

    # Block 1
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    x = Dense(512)(x)

    return Model(input1,x)
#------------------------------------------------------------------------------
def calculate_metrics(ytrue1, ypred1):
    conf = metrics.confusion_matrix(ytrue1, ypred1, [0,1])
    maxres = (conf[1,1],
              conf[0,0],
              conf[0,1],
              conf[1,0],
        metrics.precision_score(ytrue1, ypred1) * 100,
        metrics.recall_score(ytrue1, ypred1) * 100,
        metrics.f1_score(ytrue1, ypred1) * 100,
        metrics.accuracy_score(ytrue1, ypred1) * 100)
    return maxres

#------------------------------------------------------------------------------
def test_report(model_name, model, num_test_steps, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(num_test_steps):
        X, Y, paths = next(test_gen)
        Y_ = model.predict(X)
        for y1, y2, p0, p1 in zip(Y_.tolist(), Y.argmax(axis=-1).tolist(), paths[0], paths[1]):
          y1_class = np.argmax(y1)
          ypred.append(y1_class)
          ytrue.append(y2)
          a.write("%s;%s;%d;%d;%s\n" % (p0, p1, y2, y1_class, str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()

#------------------------------------------------------------------------------
def load_img(img, type, vec_size, vec_size2):
  if type is not None:
    if type=='plate':
      iplt0 = image.load_img(img[0], target_size=vec_size)
      iplt1 = image.load_img(img[2], target_size=vec_size)
    else:
      iplt0 = image.load_img(img[1], target_size=vec_size)
      iplt1 = image.load_img(img[3], target_size=vec_size)

    iplt0 = image.img_to_array(iplt0)
    iplt0 = iplt0/255.0
    iplt1 = image.img_to_array(iplt1)
    iplt1 = iplt1/255.0
    d1 = {"i0":iplt0, "i1":iplt1, "l":img[4], "p1":img[0], "p2":img[2]}

  else:
    iplt0 = image.load_img(img[0], target_size=vec_size)
    iplt1 = image.load_img(img[2], target_size=vec_size)
    iplt2 = image.load_img(img[1], target_size=vec_size2)
    iplt3 = image.load_img(img[3], target_size=vec_size2)

    iplt0 = image.img_to_array(iplt0)
    iplt0 = iplt0/255.0
    iplt1 = image.img_to_array(iplt1)
    iplt1 = iplt1/255.0
    iplt2 = image.img_to_array(iplt2)
    iplt2 = iplt2/255.0
    iplt3 = image.img_to_array(iplt3)
    iplt3 = iplt3/255.0

    d1 = {"i0":iplt0,"i1":iplt1,"i2":iplt2,"i3":iplt3,"l":img[4], "p1":img[0], "p2":img[2]}

  return d1

#------------------------------------------------------------------------------
def get_batch_inds(batch_size, idx, N):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 >= N:
            idx1 = N
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

#------------------------------------------------------------------------------
def generator(features, batch_size, type, executor, vec_size, vec_size2=None, with_paths=False):
  N = len(features)
  indices = np.arange(N)
  batchInds = get_batch_inds(batch_size, indices, N)

  while True:
    for inds in batchInds:
      futures = []
      _vec_size = (len(inds),) + vec_size
      b1 = np.zeros(_vec_size)
      b2 = np.zeros(_vec_size)
      if vec_size2 is not None:
        _vec_size2 = (len(inds),) + vec_size2
        b3 = np.zeros(_vec_size2)
        b4 = np.zeros(_vec_size2)

      blabels = np.zeros((len(inds)))
      p1 = []
      p2 = []
      futures = [executor.submit(partial(load_img, features[index], type, vec_size, vec_size2)) for index in inds]
      results = [future.result() for future in futures]

      for i,r in enumerate(results):
        b1[i,:,:,:] = r['i0']
        b2[i,:,:,:] = r['i1']
        blabels[i] = r['l']
        p1.append(r['p1'])
        p2.append(r['p2'])
        if type is None:
          b3[i,:,:,:] = r['i2']
          b4[i,:,:,:] = r['i3']

      blabels = np_utils.to_categorical(blabels, 2)
      if with_paths:
        if type is None:
            yield [b1, b2, b3, b4], blabels, [p1, p2] 
        else:
            yield [b1, b2], blabels, [p1, p2]
      else:
        if type is None:
            yield [b1, b2, b3, b4], blabels
        else:
            yield [b1, b2], blabels


#------------------------------------------------------------------------------
def run(siamese_model, type):
  data = json.load(open('dataset%d_%d.json' % (amount, multiplyNegatives)))
  trn = data['trn']
  tst = data['tst']

  np.random.shuffle(trn)

  input1 = (image_size_h_p,image_size_w_p,nchannels)
  input2 = (image_size_h_c,image_size_w_c,nchannels)

  train_steps_per_epoch = ceil(len(trn) / batch_size)
  val_steps_per_epoch = ceil(len(tst) / batch_size)


  ex1 = ProcessPoolExecutor(max_workers = 4)
  ex2 = ProcessPoolExecutor(max_workers = 4)
  ex3 = ProcessPoolExecutor(max_workers = 4)

  if type is None:
    siamese_net = siamese_model(tfs(input1), tfs(input2))
    trnGen = generator(trn, batch_size, type, ex1, input1, input2)
    tstGen = generator(tst, batch_size, type, ex2, input1, input2)
    tstGen2 = generator(tst, batch_size, type, ex3, input1, input2,True)
  elif type == 'plate':
    siamese_net = siamese_model(input1, tfs(input1))
    trnGen = generator(trn, batch_size, type, ex1, input1)
    tstGen = generator(tst, batch_size, type, ex2, input1)
    tstGen2 = generator(tst, batch_size, type, ex3, input1,None,True)
  else:
    siamese_net = siamese_model(input2, tfs(input2))
    trnGen = generator(trn, batch_size, type, ex1, input2)
    tstGen = generator(tst, batch_size, type, ex2, input2)
    tstGen2 = generator(tst, batch_size, type, ex3, input2,None,True)

  name = "two_stream" if type is None else type
  f1 = 'siamese_vehicle_%s.h5' % (name)
  c1 = ModelCheckpoint(filepath=f1, 
                      monitor='val_loss', 
                      verbose=0, 
                      save_best_only=True, 
                      save_weights_only=False, 
                      mode='auto', 
                      period=1)

  #fit model
  history = siamese_net.fit_generator(trnGen,
                                steps_per_epoch=train_steps_per_epoch,
                                epochs=NUM_EPOCHS,
                                validation_data=tstGen,
                                validation_steps=val_steps_per_epoch,
                                callbacks = [c1])
  #validate plate model
  test_report("validation_siamese_vehicle_%s" % (name),siamese_net, val_steps_per_epoch, tstGen2)
