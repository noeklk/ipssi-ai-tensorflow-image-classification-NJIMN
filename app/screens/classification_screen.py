import streamlit as st
from data_processing import *
import os

def get_directory_content(path,category):
    if category != "Choisir une catégorie":
        return os.listdir(path+'/'+category)
    else:
        return ""

def show_content(path, category, options):
    # st.write("Il existe des categories")
    for c in category:
        options.append(c)
    # st.write(options)
    selected = st.selectbox("Catégories" , options )
    list_img = get_directory_content(path, selected)
    # st.write(list_img)
    
    for i in list_img: 
        # st.write(i)
        st.image(str(os.getcwd()) + "/" +path +"/"+ selected + "/"+ i , width = 250)
    

def app():
    st.subheader("Classification")
    path = "data/app_data"
    category = os.listdir(path)
    
    len_category = len(os.listdir(path))
    options = ["Choisir une catégorie"]
    
    if len_category < 1:
        st.write("Il n'y a aucune categorie pour le moment.")
    else:
        show_content(path, category, options)
        

