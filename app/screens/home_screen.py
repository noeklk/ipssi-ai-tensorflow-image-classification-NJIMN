import streamlit as st
from data_processing import *

def app():
    st.title('Welcome')

    st.write("Vous êtes sur la page d'accueil d'une mini-app fait avec Streamlit.")
    st.write("L'objectif est d'aider un bénévole à enregistrer un chien à partir d'une photo et de le classification en fonction de sa race.")
    st.markdown("Base de données étudiée: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)")
    st.write("Cette application permet de predire une race de chien avec une photo de chien et de l'enregistrer.")
    st.write("Pour la découvrir, vous pouvez naviguer avec le menu situé sur la gauche.")
    st.markdown("## Enjoy!")


