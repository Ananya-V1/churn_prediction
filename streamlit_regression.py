import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

# Load the trained models
model = tf.keras.models.load_model('regression_model.h5')

with open('gb_regression_model.pkl', 'rb') as file:
    gb_model = pickle.load(file)

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('regression_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Estimated Salary Prediction')

# User input
model_choice = st.selectbox('Model', ['ANN', 'Gradient Boosting'])
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Predict salary
if model_choice == 'ANN':
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    predicted_salary = model.predict(input_data_scaled)[0][0]
else:
    # Gradient boosting was trained on unscaled features
    predicted_salary = gb_model.predict(input_data)[0]

st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')
