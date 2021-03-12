import streamlit as st
from data_processing import *

def app():
    st.subheader("Classification")
    races = ["Chihuahua", "Husky", "Labrador"]
    choix_race = st.selectbox("Races" ,races )
    
    if choix_race == "Chihuahua":
        st.write("Ceci est un chihuahua")
        
    elif choix_race == "Husky":
        st.write("Ceci est un husky")
        
    elif choix_race == "Labrador":
        st.write("Ceci est un labrador")

