import os
import numpy as np
import cv2
import shuffle_arrays

def open_image_dir(dir, size, label = False):
    image_pathes = os.listdir(dir)
    image_array = []
    if(label):
        flags = 0
    else:
        flags = 1
    for i in range(len(image_pathes)):
        print('loading image ', i+1, ' of ', len(image_pathes))
        img = cv2.imread(dir + '\\' + image_pathes[i], flags)
        if(size != (0,0)):
            img = cv2.resize(img, size)
        image_array.append(img)

    image_array = np.array(image_array)

    if(label):
        image_array = np.expand_dims(image_array, len(np.shape(image_array)))
        image_array = image_array / np.amax(image_array)

    return image_array

def import_images(size):
    dir_images_train = "D:\\images\\Kronos Display Images\\images"
    dir_labels_train = "D:\\images\\Kronos Display Images\\labels"
    dir_images_test = "D:\\images\\Kronos Display Images\\images test"
    dir_labels_test = "D:\\images\\Kronos Display Images\\labels test"

    train_images = open_image_dir(dir_images_train, size)
    train_labels = open_image_dir(dir_labels_train, size, label = True)
    test_images = open_image_dir(dir_images_test, size)
    test_labels = open_image_dir(dir_labels_test, size, label = True)

    train_images, train_labels = shuffle_arrays.shuffle_in_unison(train_images, train_labels)

    image_shape = np.shape(train_images[0, :, :, :])

    print("image_shape = ", image_shape)
    print("train_images shape = ", np.shape(train_images))
    print("train_labels shape = ", np.shape(train_labels))
    print("test_images shape = ", np.shape(test_images))
    print("test_labels shape = ", np.shape(test_labels))
    print("train_labels max ", np.amax(train_labels))
    print("test_labels max ", np.amax(test_labels))
    print("train_images max ", np.amax(train_images))

    return train_images, train_labels, test_images, test_labels, image_shape
