import matplotlib.pyplot as plt
import numpy as np
import cv2
from models import get_model

def visualy_inspect_generated_data(train_images, train_labels):
    plt.figure(figsize=(10, 10))
    plt.suptitle('Images with training labels', fontsize=16)
    for i in range(10):
        j = 2 * i
        plt.subplot(5, 5, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i, :, :, :], cmap=plt.cm.binary)
        plt.subplot(5, 5, j + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_labels[i, :, :, 0], cmap=plt.cm.binary)
    plt.show()

def visualy_inspect_result(train_images, image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES, loss_name):
    model = get_model(image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES)
    model.load_weights('model_weights_' + loss_name + '.h5')

    plt.figure(figsize=(10, 10))
    plt.suptitle('Images with semantic segmentation results', fontsize=16)
    for i in range(10):
        j = 2 * i
        plt.subplot(5, 5, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i, :, :, :], cmap=plt.cm.binary)
        plt.subplot(5, 5, j + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = train_images[i]
        y_pred = model.predict(img[None, ...].astype(np.float32))[0]
        plt.imshow(y_pred[:,:,0], cmap=plt.cm.binary)
    plt.show()

def plot_image(i, predictions_array, true_label, img, class_names):
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
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
