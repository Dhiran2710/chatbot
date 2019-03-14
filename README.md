# Deep Learning Models for Classification of Consumer Electronics

- `consumer_electronics_conv.py` is a convolutional neural network implemented in Tensorflow for classifying consumer electronics. 
- `inception.py` is implemented in keras and uses the state-of-the-art [inception neural network](https://arxiv.org/abs/1409.4842) and GoogLeNet for classifying consumer electronics.
- A model trained using `inception.py` is saved in the files `model.json` and `model.h5`, and can be retrieved using `retrieve_model.py`

## TODO
- Train using `inception.py` with more epochs.
- Improve accuracy of `inception.py` model. 

## To run the code on a windows machine:

- Download Python 3.6 64-bit. Note that at the time of writing, pip3 packaged with Python 3.7 encountered errors when installing Tensorflow.
- Set `PYTHONPATH` and add to `Path` environment variable the location of file `python.exe`.
- Add the location of `pip3.exe` in the `Path` environment variable, usually in `Python36/Scripts`.
- Run `pip3 install tensorflow --user`.
- Download latest version of git.  
- Add location of git to `Path` environment variable. 
- Download a text editor.
- Run `git clone https://github.com/Dhiran2710/chatbot`.
- Run `pip3 install keras --user`.
- Run `pip3 install opencv-python --user`.
- Run `pip3 install matplotlib --user`.
- Download the images for training and testing in the correct directory. The label for an image is determined by the folder it is in.
