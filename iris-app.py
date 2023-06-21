# Importing Dependencies
import streamlit as st
import numpy as np
import joblib

# Streamlit Page Configurations
st.set_page_config(page_title="Iris Flower Prediction | Aacash Srinath",
                   page_icon=None, layout="wide",
                   initial_sidebar_state="collapsed",
                   menu_items=None)

# Web App Title & Subheading
st.title("Iris Flower Prediction App")
st.subheader("Web App for Predicting the Class of Iris Flower")
st.write('')
st.write("")

# Loading the Model from the Pickle File
DTC = joblib.load('DTC.pkl')

# Slider Elements to Select Values for Given Input Fields
sepal_length = st.slider("**Sepal Length** (in cm)", 4.3, 7.9)
sepal_width = st.slider("**Sepal Width** (in cm)", 2.0, 4.4)
petal_length = st.slider("**Petal Length** (in cm)", 1.0, 6.9)
petal_width = st.slider("**Petal Width** (in cm)", 0.1, 2.5)

# Converting Input Values into a NumPy Array and Reshaping It for Prediction
pred_array = np.array((sepal_length, sepal_width, petal_length, petal_width))
pred_array = pred_array.reshape(1,-1)
st.write('')
st.write('')

# Button Element to Initiate Prediction of Given Values
pred_button = st.button("PREDICT")
if (pred_button):
    y = DTC.predict(pred_array)
    pred_val = y[0]
    st.text('')
    
    # Printing The Value of the Flower Class Predicted by the Model
    if (pred_val == 'Iris-setosa'):
        st.markdown(f"<h2 style='text-align: center;'>The Predicted Class of Flower is Iris Setosa </h2>", unsafe_allow_html=True)
    if (pred_val == 'Iris-virginica'):
        st.markdown(f"<h2 style='text-align: center;'>The Predicted Class of Flower is Iris Virginica </h2>", unsafe_allow_html=True)
    if (pred_val == 'Iris-versicolor'):
        st.markdown(f"<h2 style='text-align: center;'>The Predicted Class of Flower is Iris Versicolor </h2>", unsafe_allow_html=True)

    st.text('')
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write("")
    with col3:
        st.write("")

    # Displaying the Image of the Flower Class Predicted by the Model
    if (pred_val == 'Iris-setosa'):
        with col2:
            st.image('iris-setosa.jpg', width = 350)
    if (pred_val == 'Iris-virginica'):
        with col2:
            st.image('iris-virginica.jpg', width = 350)
    if (pred_val == 'Iris-versicolor'):
        with col2:
            st.image('iris-versicolor.jpg', width = 350)