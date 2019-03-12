# Deep Learning Models for consumer electronic classification

Deep Learning models for classifying consumer electronics. 
consumer_electronics_conv.py is a convolutional neural network for classifying consumer electronics. 
inception.py uses state-of-the-art inception neural network and GoogLeNet for classifying consumer electronics. 
A model trained using inception.py is saved in the files model.json and model.h5, and can be retrieved using retrieve_model.py

## TODO
Debug retrieve_model.py
Train with more epochs
Improve accuracy

## To run the code on a windows machine:

Download Python 3.6 64 bit (Python 3.7 didn't work for unknown reason.)
Set PYTHONPATH and Path environment variables as the location of file python.exe
Set ./Scripts/pip3.exe location as Path environment variable 
Run pip3 install tensorflow --user
Download latest version of git.  
Set the git environment variable. 
Download a text editor.
Run git clone https://github.com/Dhiran2710/chatbot
Run pip3 install keras --user
Run pip3 install opencv-python --user
Run pip3 install matplotlib --user
Download the data folder in the correct directory. The label for an image is determined by the folder it is in.