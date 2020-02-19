import keras
import numpy as np
import random

def batch_generator(train_images, train_labels, batch_size):
    while True:
        image_list = []
        mask_list = []
        k = random.sample(range(len(train_images)), batch_size)
        for i in range(batch_size):
            j = k[i]
            img, mask = train_images[j,:,:,:], train_labels[j,:,:,:]
            image_list.append(img)
            mask_list.append(mask)

        image_list = np.array(image_list,
                              dtype=np.float32)  # Note: don't scale input, because use batchnorm after input
        mask_list = np.array(mask_list, dtype=np.float32)

        yield image_list, mask_list

def import_random_images(train_images, train_labels, batch_size):
    while True:
        image_list = []
        mask_list = []
        x = random.sample(range(0, len(train_images)-1), batch_size)
        print(x)
        for i in range(batch_size):
            j = random.randint(0,len(train_images)-1)
            img, mask = train_images[j], train_labels[j]
            image_list.append(img)
            mask_list.append(mask)

        image_list = np.array(image_list, dtype=np.float32)  # Note: don't scale input, because use batchnorm after input
        mask_list = np.array(mask_list, dtype=np.float32)

        yield image_list, mask_list

'''
seed = 1337

def import_generator_data(train_images, train_labels, batch_size):
    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=5,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=1)
    image_datagen = keras.preprocessing.image.ImageDataGenerator(data_gen_args)
    mask_datagen = keras.preprocessing.image.ImageDataGenerator(data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods seed = 1
    image_datagen.fit(train_images, augment=True, seed=seed)
    mask_datagen.fit(train_labels, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        train_images,
        seed=seed,
        batch_size=batch_size)

    mask_generator = mask_datagen.flow(
        train_labels,
        seed=seed,
        batch_size=batch_size)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    return train_generator
'''
'''
    for X_batch, y_batch in mask_generator.flow(train_images, train_labels, batch_size=10):
        print("X_batch shape = ", np.shape(X_batch))
        # create a grid of 3x3 images
        for i in range(10):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(X_batch[i, :, :, :], cmap=plt.cm.binary)
            plt.xlabel("hi :-)")
        pyplot.show()
        
    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50)
        nb_epoch=50)
'''
