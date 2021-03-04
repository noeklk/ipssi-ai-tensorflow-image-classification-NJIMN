import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data_prediction
import seaborn as sns
plt.rcParams["figure.dpi"] = 140

#########################################################################################################################

# TITLE
st.markdown(
    """<link 
        rel='stylesheet' 
        href='https://use.fontawesome.com/releases/v5.8.1/css/all.css' 
        integrity='sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf' 
        crossorigin='anonymous'
    >""",
    unsafe_allow_html=True
)
st.markdown(
    '<h1><i class="fas fa-school"></i> Performance des étudiants en examen</h1>',
    unsafe_allow_html=True
)

#########################################################################################################################

st.markdown(
    '<h2>Prédiction à partir des notes</h2>',
    unsafe_allow_html=True
    )

math = st.number_input('Math', format="%d", value=0, min_value=0, max_value=100)

reading = st.number_input('Lecture', format="%d", value=0, min_value=0, max_value=100)

writing = st.number_input('Ecriture', format="%d", value=0, min_value=0, max_value=100)

data = data_prediction.notes_prediction([math, reading, writing])
data


#########################################################################################################################

st.markdown(
    '<h2>Prédiction à partir de la personne</h2>',
    unsafe_allow_html=True
    )

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

data = data_prediction.character_type_prediction([gender_output, race_output, lunch_output, parental_edu_output])
data

fig = plt.figure()
ax = fig.add_axes([0,0,1,0.5])
ax.bar(data.columns, data.loc[0], color=["red", "green", "blue"], edgecolor='black')

st.pyplot(fig)

