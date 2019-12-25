# Identifying-Appliances-from-Energy-Use-Spectrograms
This repository contains the code for the project called Identifying Application from Energy Use Spectrograms from 
[MPP Capstone Chanllenge](https://www.datasciencecapstone.org/competitions/19/appliances-energy-use-spectrograms/page/58/)
## Problem Statement
The goal of this project is to predict types of appliances from spectrograms of current and voltage measurements. The spectrograms were generated from current and voltage measurements sampled at 30 kHz from 11 different appliance types present in more than 60 households in Pittsburgh, Pennsylvania, USA. Data collection took place during the summer of 2013, and winter of 2014. Each appliance type is represented by dozens of different instances of varying make/models.
## Target Variable
The appliance labels correspond to the following appliances:

* 0: Heater
* 1: Fridge
* 2: Hairdryer
* 3: Microwave
* 4: Air Conditioner
* 5: Vacuum
* 6: Incandescent Light Bulb
* 7: Laptop
* 8: Compact Fluorescent Lamp
* 9: Fan
* 10: Washing Machine
## Data Size
The training data consists of 575 current and 575 voltage spectrograms. Each spectrogram has a size of 128 * 176. The test data consists of 383 current and 383 voltage spectrograms.
## Programming Language
Python 3.7
## Modelling
### Data Preprocessing
Data augmentation is implemented for image preprosessing. We also concatenated the current and voltage spectrograms with the same label as two channels, since there are two images as input for every instance.

The model was built as a sequential neural network, which contains eight layers, including overlapping max pooling layer and gaussian noise layer. The model is supervised by the validation accuracy to verify improvement. 

### Result
After 150 epochs of training, the validation accuracy reaches to around 0.93, ranking 4/183.


