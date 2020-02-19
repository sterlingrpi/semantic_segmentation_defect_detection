from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import get_model
from semantic_batch_generator import batch_generator
from plot_data import visualy_inspect_result
from plot_data import visualy_inspect_generated_data
from import_data import import_images
from save_data import save_prediction

# Parameters
INPUT_CHANNELS = 3
NUMBER_OF_CLASSES = 1

epochs = 10
sample_per_epoch = 50
patience = epochs
batch_size = 10

resize = (224,224) #images not resized if set to (0,0)

loss_name = "binary_crossentropy"

def train():
    model = get_model(image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('model_weights_' + loss_name + '.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit_generator(
        generator=batch_generator(train_images, train_labels, batch_size),
        nb_epoch=epochs,
        samples_per_epoch=sample_per_epoch,
        validation_data=batch_generator(train_images, train_labels, batch_size),
        nb_val_samples=10,
        verbose=1,
        shuffle=False,
        callbacks=callbacks)


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, image_shape = import_images(resize)
    visualy_inspect_generated_data(train_images, train_labels)
#    train()
    visualy_inspect_result(train_images, image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES, loss_name)
