import tensorflow as tf
from tensorflow import keras


def loadModel(model_directory):
    model = keras.models.load_model(model_directory)
    return model

def predictImage(image_directory, model_directory):
    image_size = (180, 180)
    img = keras.preprocessing.image.load_img(image_directory, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = loadModel(model_directory).predict(img_array)
    score = predictions[0]
    dogPercentage = float(100 * (1 - score))
    catPercentage = float(100 * score)
    print(catPercentage)
    return catPercentage > dogPercentage
