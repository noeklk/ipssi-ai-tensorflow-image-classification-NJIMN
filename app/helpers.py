import os
import random
import numpy as np
import pandas as pd

import cv2
cv2.ocl.setUseOpenCL(False)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler, LabelBinarizer

from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
from matplotlib import transforms

from tqdm import tqdm
import re

import tensorflow as tf

#Si vous n'avez pas de GPU ou que vous ne voulez pas l'utiliser, commentez les 2 lignes suivantes
gpus = tf.config.list_physical_devices('GPU') 

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(tf.__version__)
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing import image    
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from tensorflow.keras.models import load_model

LE = LabelEncoder()

from PIL import Image

os.environ['TF_DETERMINISTIC_OPS'] = '1' #  c'est la ligne la plus importante
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUDA_VISIBLE_DEVICES'] = '' #  c'est facultatif
os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

data_dir = './data'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test/')

race_dirs = os.listdir(train_dir)
race_dirs_indexed = LE.fit_transform(race_dirs)
class_names = pd.DataFrame([race_dirs_indexed, race_dirs]).transpose()

def all_images_path(data_type):
    all_images_path = []
    for i in race_dirs:
        folder_path = './data/' + data_type + '/' + i + '/'
        for u in os.listdir(folder_path):
            all_images_path.append(folder_path + u)
        
    return all_images_path

def get_race_from_img_path(img_path):
    full_path_img = os.path.splitext(img_path)[0]
    race = full_path_img.split('/')[3]
    index = class_names[class_names[1] == race].values[0][0]
    return index, race

ResNet50_model = ResNet50(weights='imagenet')

def img_to_tensor_trained_images(img, x, y):
    img = cv2.resize(img, (x, y))
    image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor

def ResNet50_predict_labels(img):
    img = preprocess_input(img_to_tensor_trained_images(img, 224, 224))
    return np.argmax(ResNet50_model.predict(img))

def plot_value_array_with_label(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(120))
    plt.yticks([])
    thisplot = plt.bar(x=range(120), height=predictions_array, color="#777777")
    plt.ylim([0, 0.15])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def plot_value_array_from_img(predictions_array):
    plt.grid(False)
    plt.xticks(range(120))
    plt.yticks([])
    thisplot = plt.bar(x=range(120), height=predictions_array, color="#777777")
    plt.ylim([0, 0.15])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('green')
    #thisplot[true_label].set_color('green')

def dog_detector(img):
    # Detect if an image has a dog or not
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151))

def display_img(img):
    #img = Image.load_img(img)
    img = Image.fromarray(img, 'RGB')
    
    plt.figure(figsize=(5, 5))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img):
    faces = face_cascade.detectMultiScale(img)
    return len(faces) > 0

def load_img(img_path):
    cv2.ocl.setUseOpenCL(False)
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv_rgb, gray

def format_race(race):
    return re.search(r'^[\w]{9}-(.*)$', race).group(1)











def predict_breed_from_img(cv_rgb, gray, model):
    dog_d = 'No dog found in picture'
    human_d = 'No human found in this picture'
    
    msg = [dog_d, human_d]

    predicted_label_out = 'none'

    if dog_detector(cv_rgb):
        dog_d = 'Dog found in picture'
        msg[0] = dog_d
            
        single_pred = model.predict(img_to_tensor_trained_images(cv_rgb, 150, 150), 
                                    use_multiprocessing=True)
        single_pred_label = np.argmax(single_pred)
        predicted_label = format_race(class_names[1][single_pred_label])
        
        msg.insert(0, predicted_label)
        msg.insert(1, "{:2.0f}%".format(100*np.max(single_pred), 2))

        fig, ax = plt.subplots(figsize=(25, 10))

        _ = plot_value_array_from_img(single_pred[0])
        _ = plt.xticks(range(120), [format_race(i) for i in race_dirs], rotation=90)

        predicted_label_out = predicted_label
        
        if face_detector(gray):
            human_d = 'Human found in this picture'
            msg[3] = human_d

        return cv_rgb, fig, msg, predicted_label_out
        
    elif face_detector(gray):
        human_d = 'Human found in this picture'
        msg[1] = human_d
        predicted_label_out = 'human_face'
    
    fig, ax = plt.subplots(figsize=(25, 10))
    return cv_rgb, fig, msg, predicted_label_out

encoder = LabelBinarizer()

def load_and_preprocess_image(path, x, y):
        image = cv2.imread(path)
        image = cv2.resize(image, (x,y))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
def split_data_by_env(env, shape_x, shape_y):
    BASEPATH = "./data/" + env +"/"

    LABELS = set()

    paths = []

    for d in os.listdir(BASEPATH):
        LABELS.add(d)
        paths.append((BASEPATH+d, d))

    X = []
    y = []

    for path, label in paths:
        for image_path in os.listdir(path):
            image = load_and_preprocess_image(path+"/"+image_path, shape_x, shape_y)

            X.append(image)
            y.append(label)

    X = np.array(X)
    y = encoder.fit_transform(np.array(y))
    
    return X, y, LABELS, paths

def accuracy_score_by_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test,verbose=0)
    loss_msg = f"loss on the test set is {loss:.2f}"
    acc_msg = f"accuracy on the test set is {acc:.3f}"
    return loss_msg, acc_msg

