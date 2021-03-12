import streamlit as st
from data_processing import *

def app():
    st.subheader("Comparaison des scores de précision")

    encoder = LabelBinarizer()
    X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
    st.subheader("CNN")
    model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
    loss, acc = accuracy_score_by_model(model, X_test, y_test)
    st.write(loss)
    st.write(acc)

    encoder = LabelBinarizer()
    X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
    st.subheader("VGG16")
    model = load_model('./models/VGG16_MODEL.h5')
    loss, acc = accuracy_score_by_model(model, X_test, y_test)
    st.write(loss)
    st.write(acc)