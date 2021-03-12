import streamlit as st
from data_processing import *

def app():
    st.subheader("Enregistrer un chien sur le model VGG16")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])

    category = "none"

    
    if image_file is not None:
        cv_rgb , gray = load_img_bis(image_file)
        model = load_model('./models/VGG16_MODEL.h5')
        img, fig, msg, label_out = predict_breed_from_img(cv_rgb, gray, model)

        category = label_out

        st.image(img, use_column_width=True)

        for i in msg:
            st.write(i)

        st.pyplot(fig)

        
        if st.button('Enregister ? '):
            save_img(image_file, category)

