# apps/prediction_1.py

import streamlit as st
import matplotlib.pyplot as plt
from functions.functions import character_type_prediction

plt.rcParams["figure.dpi"] = 140

def app():
    st.title('Prédiction à partir du type d\'étudiant')

    gender = st.radio(
        "Genre", 
    ("Female", "Male"))

    race = st.radio(
        "Race/Ethnicité", 
    ('Group A', 'Group B', 'Group C', 'Group D', 'Group E'))

    lunch = st.radio(
        "Repas", 
    ('Standard', 'Free/reduced'))

    parental_edu = st.radio(
        "Education parental",
        ("some high school", "high school", "associate's degree", "bachelor's degree", "master's degree")
    )

    gender_output = 0 if gender == 'Female' else 1

    race_output = 0
    if race == 'Group A':
        race_output = 0
    elif race == 'Group B':
        race_output = 1
    elif race == 'Group C':
        race_output = 2
    elif race == 'Group D':
        race_output = 3
    elif race == 'Group E':
        race_output = 4

    lunch_output = 0 if lunch == 'Free/reduced' else 1

    parental_edu_output = 0

    if parental_edu == 'associate\'s degree':
        parental_edu_output = 0
    elif parental_edu == 'bachelor\'s degree':
        parental_edu_output = 1
    elif parental_edu == 'high school':
        parental_edu_output = 2
    elif parental_edu == 'master\'s degree':
        parental_edu_output = 3
    elif parental_edu == 'some college':
        parental_edu_output = 4
    elif parental_edu == 'some high school':
        parental_edu_output = 5

    data = character_type_prediction([gender_output, race_output, lunch_output, parental_edu_output])
    st.write(data)


    fig = plt.figure()
    ax = fig.add_axes([0,0,1,0.5])
    ax.bar(data.columns, data.loc[0], color=["red", "green", "blue"], edgecolor='black')

    st.pyplot(fig)
