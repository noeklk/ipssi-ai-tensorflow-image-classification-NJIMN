import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocessing

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
    '<h1><i class="fas fa-school"></i> Student performance in exams</h1>',
    unsafe_allow_html=True
)

#########################################################################################################################

# TEST
st.markdown(
    preprocessing.test_method()
)
# TEST