import streamlit as st
from multiapp import MultiApp
from screens import home_screen , cnn_screen, vgg16_screen, accuracy_screen, predict_acc_cnn_screen, predict_acc_vgg16_screen, classification_screen


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
app.add_app("Accueil", home_screen.app)
app.add_app("Classification", classification_screen.app)
app.add_app("Modèle Custom CNN", cnn_screen.app)
app.add_app("Modèle VGG16", vgg16_screen.app)
app.add_app("Comparaison des précisions", accuracy_screen.app)
app.add_app("Predictions Precisions CNN", predict_acc_cnn_screen.app)
app.add_app("Predictions Precisions VGG16", predict_acc_vgg16_screen.app)


# The main app
app.run()
