import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

st.markdown("### About this app")
st.write("""
This tool predicts customer churn using a trained deep learning model.
It helps businesses identify high-risk customers and take proactive retention actions.
""")

st.subheader("Please input the following data so a prediction can be made:")

# User input
st.subheader("👤 Demographic Information")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)

st.markdown("---")


st.subheader("💳 Account Information")

balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')

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
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input)

# Predict churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

risk_level = (
    "High Risk 🔴" if prediction_prob > 0.7 else
    "Medium Risk 🟠" if prediction_prob > 0.4 else
    "Low Risk 🟢"
)


st.markdown("---")
st.subheader("📊 Prediction Results")

st.subheader("Probability Breakdown")

chart_data = pd.DataFrame({
    "Type": ["Churn Risk", "Retention Probability"],
    "Probability": [prediction_prob, 1 - prediction_prob]
})

st.bar_chart(chart_data.set_index("Type"))

colA, colB = st.columns(2)

with colA:
    st.metric("Churn Probability", f"{prediction_prob:.2%}")

with colB:
    st.write(risk_level)

uploaded_file = st.file_uploader("If you prefer, you may also upload a CSV for batch prediction")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)

    df["Churn Probability"] = preds

    st.write(df)