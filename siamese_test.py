from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from sys import argv
import time
import numpy as np
from keras.models import load_model
import json
from config import *
from sklearn import metrics

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


def test_report(model_name, model, num_test_steps, test_gen):
    print("=== Evaluating model: {:s} ===".format(model_name))
    a = open("%s_inferences_output.txt" % (model_name), "w")
    ytrue, ypred = [], []
    for i in range(num_test_steps):
        X, Y, paths = next(test_gen)
        Y_ = model.predict(X)
        for y1, yreg, y2, p0, p1 in zip(Y_[0].tolist(), Y_[1].tolist(), Y['class_output'].argmax(axis=-1).tolist(), paths[0], paths[1]):
          y1_class = np.argmax(y1)
          ypred.append(y1_class)
          ytrue.append(y2)
          a.write("%s;%s;%d;%d;%f;%s\n" % (p0, p1, y2, y1_class, yreg[0], str(y1)))

    a.write('tp: %d, tn: %d, fp: %d, fn: %d P:%0.2f R:%0.2f F:%0.2f A:%0.2f' % calculate_metrics(ytrue, ypred))
    a.close()


    folder = argv[2]
    for k in range(len(keys)):
      K.clear_session()
      aux = keys[:]
      aux.pop(k)
      tst = data[aux[2]] + data[aux[3]]
      ex3 = ProcessPoolExecutor(max_workers = 4)
      tst_steps_per_epoch = ceil(len(tst) / batch_size)
      tstGen2 = generator(tst, batch_size, ex3, input1, input2, with_paths = True, type='plate')
      f1 = os.path.join(folder,'model_plate_%d.h5' % (k))
      siamese_net = load_model(f1)
      test_report('test_plate_%d' % (k),siamese_net, tst_steps_per_epoch, tstGen2)

ex2 = ProcessPoolExecutor(max_workers = 8)
data = json.load(open(argv[2]))
tst = data['tst']
input1 = (image_size_h_p,image_size_w_p,nchannels)
input2 = (image_size_h_c,image_size_w_c,nchannels)

m = argv[1]
if 'car' in argv[1]:
  type = 'car'
  tstGen = generator(tst, batch_size, type, ex2, input2, None, True)
elif 'plate' in argv[1]:
  type = 'plate'
  tstGen = generator(tst, batch_size, type, ex2, input1, None, True)
else:
  type='two_stream'
  tstGen = generator(tst, batch_size, type, ex2, input1, input2, True)

w = open("validation_pred_inferences_output_%s.txt" % (type), "w")
num_test_steps = ceil(len(tst) / batch_size)
times = []
start_time = time.time()
model = load_model(m)
duration1 = time.time() - start_time
ytrue, ypred = [], []
for i in range(num_test_steps):
  X, Y, paths = next(tstGen)
  start_time = time.time()
  Y_ = model.predict(X)
  duration = time.time() - start_time
  for y1, y2, p0, p1 in zip(Y_.tolist(), Y.argmax(axis=-1).tolist(), paths[0], paths[1]):
    y1_class = np.argmax(y1)
    ypred.append(y1_class)
    ytrue.append(y2)
    w.write("%s;%s;%d;%d\n" % (p0, p1, y2, y1_class))
  times.append((duration / len(Y)))
times = np.array(times)
w.write("loadtime: %f max_time:%f min_time:%f mean_time:%f median_time:%f sum_time:%f\n" % (duration1, times.max(), times.min(), times.mean(), np.median(times), times.sum()))
w.close()
