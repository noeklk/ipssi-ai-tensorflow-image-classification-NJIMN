import streamlit as st
from multiapp import MultiApp
from screens import home, prediction_by_marks_screen, prediction_by_person_screen

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
app.add_app("Prédiction par notes", prediction_by_marks_screen.app)
app.add_app("Prédiction par type d'étudiant", prediction_by_person_screen.app)

# The main app
app.run()