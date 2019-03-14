from img_to_data import load_data
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
train_dir = './data/train/'
test_dir = './data/test/'

train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('model.h5')
print('Loaded model from disk')
initial_lrate = 0.01

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False, clipnorm=1.)
lr_sc = LearningRateScheduler(decay, verbose=1)

# evaluate loaded model on test data
loaded_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
score = loaded_model.evaluate(test_images, [test_labels, test_labels, test_labels], verbose=0)
for i in range(len(score)):
    print(loaded_model.metrics_names[i], ':', score[i])
