import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from helpers import *
import re
import glob
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelBinarizer

# File Processing Pkgs
from PIL import Image

# Load Images
def load_image(image_file):
    img = Image.open(image_file)
    return img

def display_img(img):
    imgplot = plt.imshow(img)
    imgplot = plt.yticks([])
    imgplot = plt.xticks([])
    return imgplot
    
#Charger l'image
def load_img_bis(uploaded_file):
    # Convert the file to an opencv image.
    cv2.ocl.setUseOpenCL(False)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    cv_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    return cv_rgb, gray
    
# Normaliser le nom de l'image et l'enregistrer dans le bon dossier
def save_img(uploaded_file, category):
    # Path 
    category = str(category)
    app_data_dir = "data/app_data"
    try: 
        os.makedirs(app_data_dir)
    except OSError:
        if not os.path.isdir(app_data_dir):
            raise
    path = app_data_dir+"/" + category
    # Test : Verifie si le dossier existe sinon il le
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    # Normaliser les noms des fihiers
    file_type = "jpg"
    firstname = category
    lastname = len(glob.glob(os.path.join(path, '*'))) + 1
    final_name = firstname+"_"+ str(lastname)
    with open(os.path.join(path , final_name +"." +file_type),"wb") as f: 
        f.write(uploaded_file.getbuffer())
        st.success("Saved File")

def load_and_preprocess_image(path, x, y):
        image = cv2.imread(path)
        image = cv2.resize(image, (x,y))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
def split_data_by_env(env, shape_x, shape_y, encoder):
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

    
# !!!!! A ne faire que si le dossier 'data_app' est vide
# Creer tous les dossiers de race de chien dans la bd 
# def create_all_breed():
#     nb_class = len(glob.glob(os.path.join("../data/test", '*')))
#     st.write(os.getcwd())
#     st.write(len(glob.glob(os.path.join("../data/test", '*'))))
#     for i in os.listdir("../data/test"):
#         if not i.startswith('.'):
#             # st.write(i)
#             if not os.path.isdir("/data_app/"+i):
#                 os.makedirs("data_app/"+i)
#             else:
#                 st.write("data_app/"+i+" dossier deja créé")