import glob
import itertools
import numpy as np
import os
import json
from config import *
import random
from sys import argv
verbose = False

#-----------------------------------------------------------------------------
def collecting_positive_samples (data, amount):

  #Choosing the samples:
  samples_set = []
  nsamples = 0 #Number of samples.

  for s1, s2 in data:

    #Getting the list of images:
    list1 = sorted(glob.glob(os.path.join(s1, "*")))[:amount]
    list2 = sorted(glob.glob(os.path.join(s2, "*")))[:amount]

    #Considering samples that exists for plates and vehicles of both crossings:
    if list1 == [] or list2 == []:
      continue

    #Permutation of the samples:
    for p in itertools.product(list1, list2):
      c0 = p[0].replace(plt_name, car_name)
      c1 = p[1].replace(plt_name, car_name)
      if (os.path.exists(c0)) and (os.path.exists(c1)):
        samples_set.append ((p[0], c0, p[1], c1, POS))
        nsamples += 1
        #if verbose:
        #  print (p[0], c0, p[1], c1)

    #if verbose:
    #  print ("\n\n")

  if verbose:
    print (("Number of positive samples: %d") % (nsamples))

  return samples_set

#-----------------------------------------------------------------------------
def build_positive_set (plt_set1, plt_set2, car_set1, car_set2, amount, percentage):

  #Searching for matchings labels in folder 1 against folder 2:
  nmatchings = 0 #Number of matchings.
  data = []
  for path1 in plt_set1:
    suffix = path1.split("/")[-1]
    #Labels from license plate annotations (cross1).
    #Trying to find the same labels in cross1 and 2 for license plate and vehicle shapes.
    path2 = list(filter(lambda x: suffix in x, plt_set2))
    if path1 != [] and path2 != []:
      data.append((path1, path2[0]))
      nmatchings += 1

  if (verbose):
    print (("Number of vehicles matchings: %d") % (nmatchings))

  #Dataset size of the positive samples:
  data_size = len(data)

  #Shuffling images to use for training and testing different day periods:
  np.random.shuffle(data)

  #Spliting the testing and training samples:
  ptrn_data = data[ : int(data_size * percentage)]  #Trainning set
  ptst_data = data[int(data_size * percentage) : ]  #Testing set

  #Adding the images:
  ptrn_set = collecting_positive_samples (ptrn_data, amount)
  ptst_set = collecting_positive_samples (ptst_data, amount)

  return ptrn_set, ptst_set

#-----------------------------------------------------------------------------
def distance_string (ocr1, ocr2):
  return sum (ocr1[i] != ocr2[i] for i in range(min(len(ocr1),len(ocr2))) )

#-----------------------------------------------------------------------------
def collecting_negative_samples (plt_set1, plt_set2, car_set1, car_set2, nsamples, multiply):
  #Choosing the samples:
  labels_set = []
  samples_set = []
  failed = 0
  tries = 0
  threshold = 1000
  samples = 0
  percentage = int(nsamples/7);              #Samples distribution
  hist = [0, 0, 0, 0, 0, 0, 0, 0]            #Histogram to accumulate the characters
  while (samples < nsamples):
    r1 = np.random.choice(plt_set1)
    r2 = np.random.choice(plt_set2)
    #if verbose:
    #  print ("negative",r1, r2)
    n1 = r1.split("/")[-1]
    n2 = r2.split("/")[-1]
    p1 = list(filter(lambda x: n1 in x, car_set1)) #Matchings
    p2 = list(filter(lambda x: n2 in x, car_set2)) #Matchings

    dist = int(distance_string(n1, n2))
    if (hist[dist] < percentage or tries > threshold):
      if (n1 != []) and (n2 != []) and (p1 != []) and (p2 != []) and (n1 != n2) and ((n1,n2) not in labels_set) and ((n2,n1) not in labels_set):
         plt0 = random.choice(glob.glob(os.path.join(r1,"*")))
         plt1 = random.choice(glob.glob(os.path.join(r2,"*")))
         car0 = plt0.replace(plt_name, car_name)
         car1 = plt1.replace(plt_name, car_name)
         if os.path.exists(car0) and os.path.exists(car1):
           labels_set.append((n1,n2))
           labels_set.append((n2,n1))
           samples_set.append ((plt0, car0, plt1, car1, NEG))
           samples += 1
           hist[dist] += 1
           #if verbose:
           #  print ("negative join", plt0, plt1, car0, car1)
      else:
         failed += 1
    else:
      tries += 1
  print (("Number of samples rejected: %d") % (failed))
  if verbose:
    print (hist)

  return samples_set

#-----------------------------------------------------------------------------
def build_negative_set (plt_set1, plt_set2, car_set1, car_set2, ptrn_set, ptst_set, amount, multiply):

  size = len(ptrn_set) + multiply * len(ptst_set)
  data = collecting_negative_samples (plt_set1, plt_set2, car_set1, car_set2, size, amount)
  #Shuffling images to use for training and testing different day periods:
  np.random.shuffle(data)

  #Spliting the testing and training samples:
  ntrn_set = data[ : len(ptrn_set)]  #Trainning set
  ntst_set = data[len(ptrn_set) : ]  #Testing set

  return ntrn_set, ntst_set

def run(plt_set1, plt_set2, car_set1, car_set2, amount, percentage, multiply):
    #Building the positive training and testing datasets:
    pos_trn_set, pos_tst_set = build_positive_set (plt_set1, plt_set2, car_set1, car_set2, amount, percentage)
    #Building the negatives training and testing datasets:
    neg_trn_set, neg_tst_set = build_negative_set (plt_set1, plt_set2, car_set1, car_set2, pos_trn_set, pos_tst_set, amount, multiply)

    trn = pos_trn_set + neg_trn_set
    tst = pos_tst_set + neg_tst_set

    dataset = {'trn':trn,'tst':tst}
    return {"dataset":dataset,"amount":amount, "multiply":multiply}

folder_cross1 = 'dataset2/Cruzamento1'
folder_cross2 = 'dataset2/Cruzamento2'
plt_name="classes"
car_name="classes_carros"
plt_folder="*/classes"
car_folder="*/classes_carros"
percentage = 0.5

plt_set1 = glob.glob(os.path.join(folder_cross1, plt_folder,"*")) #License plate path for images in crossing 1.
plt_set2 = glob.glob(os.path.join(folder_cross2, plt_folder,"*")) #License plate path for images in crossing 2.
car_set1 = glob.glob(os.path.join(folder_cross1, car_folder,"*")) #Vehicle path for images in crossing 1.
car_set2 = glob.glob(os.path.join(folder_cross2, car_folder,"*")) #Vehicle path for images in crossing 2.

r = run(plt_set1, plt_set2, car_set1, car_set2, amount, percentage, multiplyNegatives)
with open('dataset%d_%d.json' % (r['amount'], r['multiply']), 'w') as fp:
  json.dump(r['dataset'], fp)
