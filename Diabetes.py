#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


# In[3]:


model = load_model('Diabetes_Prediction_India')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo_pycaret.png')
    #image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict Diabetes among women in India')
    st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    st.title("Diabetes Prediction App")

    if add_selectbox == 'Online':

        Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=100, value=25)
        Glucose = st.number_input('Glucose', min_value=0, max_value=500000, value=25)
        BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=500000, value=25)
        SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=500000, value=25)
        Insulin = st.number_input('Insulin', min_value=0, max_value=500000, value=25)
        #sex = st.selectbox('Sex', ['male', 'female'])
        BMI = st.number_input('BMI', min_value=0.0, max_value=5000.0, value=10.0)
        DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=5000.0, value=10.0)
        Age = st.number_input('Age', min_value=0, max_value=5000, value=10)
        #children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        #smoker = st.selectbox('smoker', ['yes', 'no'])
        
       # region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'Pregnancies' : Pregnancies, 'Glucose' : Glucose, 'BloodPressure' : BloodPressure, 'SkinThickness' : SkinThickness, 
                      'Insulin' : Insulin, 'BMI' : BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()


# In[ ]:




