import streamlit as st
import matplotlib.pyplot as plt
from data_processing import notes_prediction

def app():
    st.title('Prédiction à partir des notes')

    math = st.number_input('Math', format="%d", value=0, min_value=0, max_value=100)

    reading = st.number_input('Lecture', format="%d", value=0, min_value=0, max_value=100)

    writing = st.number_input('Ecriture', format="%d", value=0, min_value=0, max_value=100)

    data = notes_prediction([math, reading, writing])
    
    st.write(data)
