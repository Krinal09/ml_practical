import streamlit as st
import pandas as pd
import pickle

with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Linear Regression Model Prediction")
st.write("Enter the features to get a prediction from the Linear Regression model.")

st.sidebar.header("Input Features")

def user_input_features():  
    OverallQual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    GrLivArea = st.sidebar.slider('Ground Living Area', 0, 4000, 1500)
    GarageCars = st.sidebar.slider('Garage Cars', 0, 4, 2)
    TotalBsmtSF = st.sidebar.slider('Total Basement SF', 0, 5000, 1000)
    YearBuilt = st.sidebar.slider('Year Built', 1900, 2022, 2000)

    data = {
        'OverallQual': OverallQual,   
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF,
        'YearBuilt': YearBuilt
    }
    features = pd.DataFrame(data, index=[0])    
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

prediction = model.predict(input_df)
    
st.subheader('Prediction')
st.write(f'Predicted Sale Price: {prediction[0]:,.2f}')
    
