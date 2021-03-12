import streamlit as st
from data_processing import *

def app():
    st.subheader("Predictions Precisions CNN")

    encoder = LabelBinarizer()
    X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
    st.subheader("CNN")
    model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
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

