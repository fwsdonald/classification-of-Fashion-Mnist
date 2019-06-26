#coding=utf-8

import numpy as np
import functools
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.top2_acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.val_top2_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.top2_acc['batch'].append(logs.get('top2_acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        self.val_top2_acc['batch'].append(logs.get('val_top2_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.top2_acc['epoch'].append(logs.get('top2_acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.val_top2_acc['epoch'].append(logs.get('val_top2_acc'))

    def loss_plot(self, loss_type , model_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        # top2_acc
        plt.plot(iters, self.top2_acc[loss_type], 'c', label='train top2 acc')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            # val_top2_acc
            plt.plot(iters, self.val_top2_acc[loss_type], 'm', label='val top2 acc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="center right")
        plt.show()
        plt.savefig('/home/user045/fws/JQXX/Fashion_MNIST_Classification/result/'+model_type+'/Loss_Acc_Curve.png')

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
train_images = train_images.reshape([-1,28,28]) / 255.0
test_images = test_images.reshape([-1,28,28]) / 255.0

# Preprocessing
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Model

model = keras.Sequential([
    #(-1,28,28)->(-1,100)
    keras.layers.SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    input_shape=(28, 28),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    units=512,
    unroll=True),
    keras.layers.Dropout(rate=0.5),
    #(-1,256)->(-1,10)
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# compile the model
model.summary()

#top2
top2_acc = functools.partial(keras.metrics.sparse_top_k_categorical_accuracy, k=2)
top2_acc.__name__ = 'top2_acc'

model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy', top2_acc]) 

# train the model

history = LossHistory()
model.fit(train_images,train_labels, epochs = 100,validation_data=[test_images[:1000],test_labels[:1000]],callbacks=[history])
test_score = model.evaluate(test_images, test_labels)

# evaluate the model
print('top1_Test_accuracy:', test_score[1], 'top2_Test_accuracy:', test_score[2])

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
plt.savefig('/home/user045/fws/JQXX/Fashion_MNIST_Classification/result/RNN/Prediction.png')
history.loss_plot('epoch','RNN')