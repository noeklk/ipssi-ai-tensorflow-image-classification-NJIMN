import streamlit as st
from multiapp import MultiApp
from apps import home, prediction_by_marks, prediction_by_person

st.markdown(
    """<link 
        rel='stylesheet' 
        href='https://use.fontawesome.com/releases/v5.8.1/css/all.css' 
        integrity='sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf' 
        crossorigin='anonymous'
    >""",
    unsafe_allow_html=True
)


app = MultiApp()

# Add all your application here
app.add_app("Accueil", home.app)
app.add_app("Prédiction par notes", prediction_by_marks.app)
app.add_app("Prédiction par type d'étudiant", prediction_by_person.app)

# The main app
app.run()