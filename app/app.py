import streamlit as st

# File Processing Pkgs
from PIL import Image

# Load Images
def load_image(image_file):
    img = Image.open(image_file)
    return img
    
    
def main():
    st.title("My File upload app")
    
    menu =["Home", "Dataset" , "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload Images" , type=["png", "jpg" , "jpeg"])
        if image_file is not None:
            # To see details
            st.write(type(image_file))
            # Methods & Attributs
            # st.write(dir(image_file))
            file_details = {"filename" : image_file.name,
                            "filetype" : image_file.type,
                            "filesize" : image_file.size}
            st.write(file_details)
            
            st.image(load_image(image_file),width=250)
    

        
if __name__ == '__main__':
    main()