import time
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import numpy as np
from datasets import convert
import random as rn
import argparse
import cv2

from decimal import Decimal, ROUND_HALF_UP
from models import get_model_config, get_model_name, load_model

def predict(model_config, file_location):
    model = load_model(model_config['filepath_weight'], model_config['filepath_architechture'])

    # Convert images to correct format
    a = []
    img = skio.imread(file_location)
    img = resize(img, (16, 8))
    img = img.tolist()
    a.append(img)
    img = np.asarray(a)
    x_test = img

    # Finds confidence of all 26 alphabets
    prediction = model.predict(x_test, batch_size=32, verbose=0)
    result = np.argmax(prediction, axis=1)
    result = result.tolist()
    for i in prediction:
        confidence = prediction[0][result]

    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    confidence= Decimal(confidence[0]*100)

    confidence = Decimal(confidence.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return result_alphabet[0], confidence


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        print('\t%s comsumed %.1fs' % (func.__name__, time.time() - started_at))
        return result
    return wrap

def answer_convert(result_alphabet, index_result):
    """
    Input the test data with the trained model and tags the predicted label with the id
    """
    result_list = [(index_result[i], result_alphabet[i]) for i in range(len(index_result))]

    # Tags the train label with thier index
    array, label, index_train = convert('train')
    label_alphabet = [chr(int(i) + ord('a')) for i in label]
    train_list = [(index_train[i], label_alphabet[i]) for i in range(len(index_train))]

    # Final sorting
    final_answer_for_submission = result_list + train_list
    final_answer_for_submission = sorted(final_answer_for_submission)

    return final_answer_for_submission

def get_image_samples(n, show = False, save = True):
    array, label, index = convert('test')
    images = []

    positions = rn.sample(range(len(array)), n)
    for position in positions:
        images.append(array[position])

    for i, img in enumerate(images):
        img = np.asarray(img, dtype = np.float64)
        if show:
            skio.imshow(img)
            plt.show()
        if save:
            filepath = './data/show_image{}.jpg'.format(i)
            img = rescale(img, 4)
            skio.imsave(filepath, img)

def ask_for_file_particulars():
    parser = argparse.ArgumentParser(description="Input file location of image to test with trained model.")
    parser.add_argument("location", help="Location of the image")
    # parser.add_argument('length', help ='Length of the image', type = int)
    # parser.add_argument('width', help ='Width of the image', type = int)
    args = parser.parse_args()
    # size = (args.length, args.width)
    return args.location


#parameter 'img' should pe passes as { img = cv2.imread('coins.png') }
def get_white_foreground_and_black_background(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh, bnw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite('./../output/bnw.png', bnw)
    
if __name__ == '__main__':
    get_image_samples(3)
