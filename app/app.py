import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from helpers import *

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
    

def load_img_bis(uploaded_file):
    # Convert the file to an opencv image.
    cv2.ocl.setUseOpenCL(False)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    cv_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    return cv_rgb, gray
    
def save_img(uploaded_file, path):
    # Test : Verifie si le dossier existe sinon il le
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    with open(os.path.join(path , uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())
        st.success("Saved File")
    
    
def main():
    st.title("Mini app")
    
    menu =["Home", "Dataset" , "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])
        
        
        # TEST : on suppose qu'il prevoit que le chien est un chihuahua
        # To do: récupérer de facon dynamique la race du chien
        dogs_breed = "n02085620-Chihuahua"
        
        # Path 
        path = "data/"+dogs_breed

        if image_file is not None:
            cv_rgb , gray = load_img_bis(image_file)
            #st.image(cv_rgb)
            #st.write(image_file.name)


            
            
            if st.button('Enregister? '):
                save_img(image_file, path)
                
                
                
                
            
            
if __name__ == '__main__':
    main()