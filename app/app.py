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
app.add_app("Model Custom CNN", cnn_screen.app)
app.add_app("Model VGG16", vgg16_screen.app)
app.add_app("Comparaison des précisions", accuracy_screen.app)
app.add_app("Predictions Precisions CNN", predict_acc_cnn_screen.app)
app.add_app("Predictions Precisions VGG16", predict_acc_vgg16_screen.app)
app.add_app("Classification", classification_screen.app)

# The main app
app.run()






# # Visualisation de l'app
# def main():
#     st.title("Mini app")
    
#     menu =["Home", "CNN", "VGG16", "Accuracy Scores", "Predictions Precisions CNN", "Predictions Precisions VGG16"]
#     choice = st.sidebar.selectbox("Menu", menu)
    
#     if choice == "Home":
#         st.subheader("Home")
#         st.markdown("# Hello!")
    
#     elif choice == "CNN":
#         st.subheader("Enregistrer un chien sur le model custom CNN")
#         image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])
        
#         category = "none"
      
#         if image_file is not None:
#             cv_rgb , gray = load_img_bis(image_file)
#             model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
#             img, fig, msg, label_out = predict_breed_from_img(cv_rgb, gray, model)

#             category = label_out

#             st.image(img, use_column_width=True)

#             for i in msg:
#                 st.write(i)

#             st.pyplot(fig)

            
#             if st.button('Enregister? '):
#                 save_img(image_file, category)

#     elif choice == "VGG16":
#         st.subheader("Enregistrer un chien sur le model VGG16")
#         image_file = st.file_uploader("Upload Image", type=["png", "jpg" , "jpeg"])

#         category = "none"

      
#         if image_file is not None:
#             cv_rgb , gray = load_img_bis(image_file)
#             model = load_model('./models/VGG16_MODEL.h5')
#             img, fig, msg, label_out = predict_breed_from_img(cv_rgb, gray, model)

#             category = label_out

#             st.image(img, use_column_width=True)

#             for i in msg:
#                 st.write(i)

#             st.pyplot(fig)

            
#             if st.button('Enregister? '):
#                 save_img(image_file, category)

#     elif choice == "Accuracy Scores":
#         st.subheader("Accuracy Score Comparison")

#         encoder = LabelBinarizer()
#         X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
#         st.subheader("CNN")
#         model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
#         loss, acc = accuracy_score_by_model(model, X_test, y_test)
#         st.write(loss)
#         st.write(acc)

#         encoder = LabelBinarizer()
#         X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
#         st.subheader("VGG16")
#         model = load_model('./models/VGG16_MODEL.h5')
#         loss, acc = accuracy_score_by_model(model, X_test, y_test)
#         st.write(loss)
#         st.write(acc)
        
#     elif choice == "Predictions Precisions CNN":
#         st.subheader("Predictions Precisions CNN")

#         encoder = LabelBinarizer()
#         X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 150, 150, encoder)
#         st.subheader("CNN")
#         model = load_model('./models/dogs_dataset_big_50_batch_2.h5')
#         predictions = model.predict(X_test)
#         label_predictions = encoder.inverse_transform(predictions)

#         rows, cols = 3, 3
#         size = 25

#         fig,ax=plt.subplots(rows,cols)
#         fig.set_size_inches(size,size)
#         for i in range(rows):
#             for j in range (cols):
#                 index = np.random.randint(0,len(X_test))
#                 ax[i,j].imshow(X_test[index])
#                 ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

#         st.pyplot(fig)

#     elif choice == "Predictions Precisions VGG16":
        # st.subheader("Predictions Precisions VGG16")

        # encoder = LabelBinarizer()
        # X_test, y_test, LABELS_test, path_test = split_data_by_env('test', 224, 224, encoder)
        # st.subheader("VGG16")
        # model = load_model('./models/VGG16_MODEL.h5')
        # predictions = model.predict(X_test)
        # label_predictions = encoder.inverse_transform(predictions)

        # rows, cols = 3, 3
        # size = 25

        # fig,ax=plt.subplots(rows,cols)
        # fig.set_size_inches(size,size)
        # for i in range(rows):
        #     for j in range (cols):
        #         index = np.random.randint(0,len(X_test))
        #         ax[i,j].imshow(X_test[index])
        #         ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')

        # st.pyplot(fig)

    # elif choice == "Classification":
    #     st.subheader("Classification")
    #     races = ["Chihuahua", "Husky", "Labrador"]
    #     choix_race = st.selectbox("Races" ,races )
        
    #     if choix_race == "Chihuahua":
    #         st.write("Ceci est un chihuahua")
            
    #     elif choix_race == "Husky":
    #         st.write("Ceci est un husky")
            
    #     elif choix_race == "Labrador":
    #         st.write("Ceci est un labrador")
            
        

# if __name__ == '__main__':
#     main()


