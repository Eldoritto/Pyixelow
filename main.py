from tensorflow import keras
from PIL import Image
import io


def loadModel(model_directory):
    model = keras.models.load_model(model_directory)
    return model

def predictImage(image_directory, model_directory):
    image_size = (180, 180)
    img = Image.open(io.BytesIO(image_directory))
    img = img.convert('RGB')
    img = img.resize(image_size, Image.NEAREST)
    img = Image.img_to_array(img)
    predictions = loadModel(model_directory).predict(img)
    score = predictions[0]
    dogPercentage = float(100 * (1 - score))
    catPercentage = float(100 * score)

    return catPercentage > dogPercentage
