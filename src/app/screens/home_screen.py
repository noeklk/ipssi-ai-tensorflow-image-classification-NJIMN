import streamlit as st
import pandas as pd
import numpy as np

def app():
    st.title('Welcome HumanBot')

    st.write("Vous êtes sur la page d'accueil d'une mini-app fait avec Streamlit.")
    st.write("L'objectif est de découvrir les bases de l'IA et de prédire des résultats potentiels à partir d'une base de données.")
    st.markdown("Base de données étudiée: [Students Performance](https://www.kaggle.com/spscientist/students-performance-in-exams)")
    st.write("Cette application permet de faire des prédictions suivants différents critères:")
    st.markdown("- Prédiction par notes")
    st.markdown("- Prédictions par type d'étudiant")
    
    st.write("Pour découvrir les résultats, vous pouvez naviguer avec le menu situé sur la gauche.")
    st.markdown("## Enjoy!")

