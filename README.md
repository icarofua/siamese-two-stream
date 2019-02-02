# siamese-two-stream
We describe in this paper a novel Two-Stream Siamese Neural Network for vehicle re-identification. The proposed network is fed simultaneously with small coarse 
patches of the vehicle shape's, with 96 x 96 pixels, in one stream, and fine features extracted from license plate patches, easily readable by humans, 
with 96 x 48 pixels, in the other one. Then, we combined the strengths of both streams by merging the siamese distance descriptors with a sequence of 
fully connected layers, as an attempt to tackle a major problem in the field, false alarms caused by a huge number of car design and models with nearly the same 
appearance or by similar license plate strings. In our experiments, with 2 hours of videos containing 2982 vehicles, extracted from two low-cost cameras in the 
same roadway, 546 ft away, we achieved a $F$-measure and accuracy of 92.6% and 98.7%, respectively. 
We show that the proposed network outperforms other One-Stream architectures, even if they use higher resolution image features.

## installation
pip install keras tensorflow scikit-learn futures functools

## generate dataset
python generate_dataset.py

## siamese_plate
python siamese.py plate

## siamese_car
python siamese.py car

## siamese_fusion
python siamese_two_stream.py

## predict
python siamese_pred.py siamese_vehicle_two_stream.h5
