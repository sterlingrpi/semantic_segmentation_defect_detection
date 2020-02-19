import os
from PIL import Image
from models import get_model
import numpy as np
import cv2

def save_prediction(train_images, train_labels, image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES, loss_name):
    model = get_model(image_shape, INPUT_CHANNELS, NUMBER_OF_CLASSES)
    model.load_weights('model_weights_' + loss_name + '.h5')
    IMAGE_H, IMAGE_W = image_shape[0], image_shape[1]

    i = 0
    img, mask = train_images[i], train_labels[i]

    y_pred = model.predict(img[None, ...].astype(np.float32))[0]

    print('y_pred.shape', y_pred.shape)

    y_pred = y_pred.reshape((IMAGE_H, IMAGE_W, NUMBER_OF_CLASSES))

    print('np.min(mask[:,:,0])', np.min(mask[:, :, 0]))
    print('np.max(mask[:,:,0])', np.max(mask[:, :, 0]))

    print('np.min(y_pred)', np.min(y_pred))
    print('np.max(y_pred)', np.max(y_pred))

    res = np.zeros((IMAGE_H, 4 * IMAGE_W, 3), np.uint8)
    res[:, :IMAGE_W, :] = img
    print("shape of mask b4 res", np.shape(mask))
    print("max of mask", np.amax(mask))
    mask = mask*255
    res[:, IMAGE_W:2 * IMAGE_W, :] = cv2.cvtColor(mask[:, :, 0], cv2.COLOR_GRAY2RGB)
    res[:, 2 * IMAGE_W:3 * IMAGE_W, :] = 255 * cv2.cvtColor(y_pred[:, :, 0], cv2.COLOR_GRAY2RGB)
    y_pred[:, :, 0][y_pred[:, :, 0] > 0.5] = 255
    res[:, 3 * IMAGE_W:4 * IMAGE_W, :] = cv2.cvtColor(y_pred[:, :, 0], cv2.COLOR_GRAY2RGB)

    cv2.imwrite(loss_name + '_result.png', res)

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)

if __name__ == '__main__':
    base_dirs = [
        'C:\\Users\\snesbitt\\Desktop\\Kronos Display Images\\images',
        'C:\\Users\\snesbitt\\Desktop\\Kronos Display Images\\labels',
        'C:\\Users\\snesbitt\\Desktop\\Kronos Display Images\\images test',
        'C:\\Users\\snesbitt\\Desktop\\Kronos Display Images\\labels test']
    for j in range(0, len(base_dirs)):
        base_dir = base_dirs[j]
        image_path = os.listdir(base_dir)
        for i in range(0, len(image_path)):
            resize_image(base_dir + '\\' + image_path[i], base_dir + '\\' + 'resized' + image_path[i], (224, 224))
