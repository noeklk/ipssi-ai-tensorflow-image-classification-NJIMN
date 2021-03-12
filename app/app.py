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
    path = "data/app_data/" + category
    # Test : Verifie si le dossier existe sinon il le
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    # Normaliser les noms des fihiers
    file_type = "jpg"
    # modele = n02085620_477.jpg
    # nom du dossier = n02085620_Chihuahua
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


# Visualisation de l'app
def main():
    st.title("Mini app")
    
    menu =["Home", "CNN", "VGG16", "Accuracy Scores", "Predictions Precisions CNN", "Predictions Precisions VGG16"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        st.markdown("# Hello!")
    
    elif choice == "CNN":
        st.subheader("Enregistrer un chien sur le model custom CNN")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])
        
        category = "none"
      
        if image_file is not None:
            cv_rgb , gray = load_img_bis(image_file)
            model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
            img, fig, msg, label_out = predict_breed_from_img(cv_rgb, gray, model)

            category = label_out

            st.image(img, use_column_width=True)

            for i in msg:
                st.write(i)

            st.pyplot(fig)

            
            if st.button('Enregister? '):
                save_img(image_file, category)

    elif choice == "VGG16":
        st.subheader("Enregistrer un chien sur le model VGG16")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])

        category = "none"

      
        if image_file is not None:
            cv_rgb , gray = load_img_bis(image_file)
            model = load_model('./models/VGG16_MODEL.h5')
            img, fig, msg, label_out = predict_breed_from_img(cv_rgb, gray, model)

            category = label_out

            st.image(img, use_column_width=True)

            for i in msg:
                st.write(i)

            st.pyplot(fig)

            
            if st.button('Enregister? '):
                save_img(image_file, category)

    elif choice == "Accuracy Scores":
        st.subheader("Accuracy Score Comparison")

        encoder = LabelBinarizer()
        X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
        st.subheader("CNN")
        model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
        loss, acc = accuracy_score_by_model(model, X_test, y_test)
        st.write(loss)
        st.write(acc)

        encoder = LabelBinarizer()
        X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
        st.subheader("VGG16")
        model = load_model('./models/VGG16_MODEL.h5')
        loss, acc = accuracy_score_by_model(model, X_test, y_test)
        st.write(loss)
        st.write(acc)
        
    elif choice == "Predictions Precisions CNN":
        st.subheader("Predictions Precisions CNN")

        encoder = LabelBinarizer()
        X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
        st.subheader("CNN")
        model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
        predictions = model.predict(X_test)
        label_predictions = encoder.inverse_transform(predictions)

        rows, cols = 3, 3
        size = 25

        fig,ax=plt.subplots(rows,cols)
        fig.set_size_inches(size,size)
        for i in range(rows):
            for j in range (cols):
                index = np.random.randint(0,len(X_test))
                ax[i,j].imshow(X_test[index])
                ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

        st.pyplot(fig)

    elif choice == "Predictions Precisions VGG16":
        st.subheader("Predictions Precisions VGG16")

        encoder = LabelBinarizer()
        X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
        st.subheader("VGG16")
        model = load_model('./models/VGG16_MODEL.h5')
        predictions = model.predict(X_test)
        label_predictions = encoder.inverse_transform(predictions)

        rows, cols = 3, 3
        size = 25

        fig,ax=plt.subplots(rows,cols)
        fig.set_size_inches(size,size)
        for i in range(rows):
            for j in range (cols):
                index = np.random.randint(0,len(X_test))
                ax[i,j].imshow(X_test[index])
                ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

        st.pyplot(fig)

    # elif choice == "Classification":
    #     st.subheader("Classification")
    #     races = ["Chihuahua", "Husky", "Labrador"]
    #     choix_race = st.selectbox("Races" ,races )
        
    #     if choix_race == "Chihuahua":
    #         st.write("Ceci est un chihuahua")
            
    #     elif choix_race == "Husky":
    #         st.write("Ceci est un husky")
            
    #     elif choix_race == "Labrador":
    #         st.write("Ceci est un labrador")
            

        

if __name__ == '__main__':
    main()



