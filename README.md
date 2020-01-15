# Deprecated: see (https://github.com/icarofua/vehicle-ReId) in order to have the full dataset and new models.
# Demo code for paper "A Two-Stream Siamese Neural Network for Vehicle Re-Identification by Using Non-Overlapping Cameras" (https://arxiv.org/abs/1902.01496).



If you find this code useful in your research, please consider citing:

    @article{icaroICIP2019,
        title={A Two-Stream Siamese Neural Network for Vehicle Re-Identification by Using Non-Overlapping Cameras},
        author={de Oliveira, Icaro O and Fonseca, Keiko VO and Minetto, Rodrigo},
        journal={IEEE International Conference on Image Processing (ICIP)},
        year={2019}
    }

## Authors

- Ícaro Oliveira de Oliveira
- Keiko Veronica Ono Fonseca
- Rodrigo Minetto

## 1. Installation of the packages
pip3 install keras tensorflow scikit-learn futures

## 2 Configuration
config.py

## 3.1 training of the siamese plate
python3 siamese.py plate

## 3.2 training of the siamese car
python3 siamese.py car

## 3.3 training of the siamese two stream
python3 siamese_two_stream.py

## 4. testing the model with generated dataset or other dataset following the format in 3.2.
python3 siamese_test.py siamese_original_two_stream.h5 dataset1_10.json

## 5. generate dataset.
python3 generate_dataset.py
