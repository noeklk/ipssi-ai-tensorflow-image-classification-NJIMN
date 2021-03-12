import streamlit as st
from data_processing import *

def app():
    st.subheader("Predictions Precisions VGG16")

    encoder = LabelBinarizer()
    X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
    st.subheader("VGG16")
    model = load_model('./models/VGG16_MODEL.h5')
    predictions = model.predict(X_test)
    label_predictions = encoder.inverse_transform(predictions)

    rows, cols = 3, 3
    size = 25

    fig,ax=plt.subplots(rows,cols)
    fig.set_size_inches(size,size)
    for i in range(rows):
        for j in range (cols):
            index = np.random.randint(0,len(X_test))
            ax[i,j].imshow(X_test[index])
            ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

    st.pyplot(fig)

