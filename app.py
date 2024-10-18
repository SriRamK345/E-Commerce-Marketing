import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import joblib  # If you saved your model using joblib



# Load your trained models (replace with your actual model loading)
# Assuming you saved using pickle:

lr_model_c = joblib.load("E-Comm sales/lr_model_camera.pkl")

lr_model_a = joblib.load("E-Comm sales/lr_model_audio.pkl")

lr_model_g = joblib.load("E-Comm sales/lr_model_gaming.pkl")

# Load the pre-fitted scaler
# scaler = joblib.load("E-Comm sales/scaler.pkl")  # Adjust the path as necessary

# Define the expected feature names based on training

# Function for preprocessing input data
def preprocess_data(input_data, sub_category):
    """
    Applies preprocessing steps based on sub-category.
    """
    #  One-Hot Encoding
    if sub_category == "CameraAccessory":
        # Apply one-hot encoding for CameraAccessory
         input_data = pd.get_dummies(input_data, 
                                      columns=["product_analytic_sub_category_CameraAccessory"], 
                                      drop_first=True)
    if sub_category == "HomeAudio":
         input_data = pd.get_dummies(input_data, 
                                      columns=["product_analytic_sub_category_HomeAudio"], 
                                    drop_first=True)
                        
    elif sub_category == "GamingAccessory":
         input_data = pd.get_dummies(input_data, 
                                      columns=["product_analytic_sub_category_GamingAccessory"], 
                                      drop_first=True)
    


    # sqrt
    input_data["total_units_sqrt"] = np.sqrt(input_data["total_units_sqrt"])

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))  # Or load your pre-trained scaler
    numerical_cols = input_data.select_dtypes(include=["int64", "float64"]).columns
    input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])
    
    return input_data

# Streamlit app structure
st.title("E-Commerce Sales Prediction")

# Product Sub-category Selection
sub_category = st.selectbox("Select Product Sub-category", ["CameraAccessory", "HomeAudio", "GamingAccessory"])

# Input fields for features (customize based on your features)
units = st.number_input("Enter Total Units", value=0.0)
sla = st.number_input("Enter SLA", value=0.0)
avg_temperature = st.number_input("Enter Average Temperature", value=0.0)
average_order_value = st.number_input("Enter Average Order Value", value=0.0)
pay_date_flag = st.selectbox("Pay Date Flag", [0, 1])
holiday_flag = st.selectbox("Holiday Flag", [0, 1])

# Prediction button
if st.button("Predict"):

    # Create input data DataFrame
    input_data = pd.DataFrame({
        "total_units_sqrt": [units],
        "sla": [sla],
        "avg_temperature": [avg_temperature],
        "Average_Order_Value": [average_order_value],
        "pay_date_flag": [pay_date_flag],
        "holiday_flag": [holiday_flag],
        "product_analytic_sub_category_CameraAccessory": [1 if sub_category == "CameraAccessory" else 0],
        "product_analytic_sub_category_HomeAudio": [1 if sub_category == "HomeAudio" else 0],
        "product_analytic_sub_category_GamingAccessory": [1 if sub_category == "GamingAccessory" else 0]
    })

    # Select the appropriate model based on sub-category
    if sub_category == "CameraAccessory":
        model = lr_model_c
    elif sub_category == "HomeAudio":
        model = lr_model_a
    else:
        model = lr_model_g

    # Preprocess the input data
    input_data_processed = preprocess_data(input_data, sub_category)

    try:
        prediction = model.predict(input_data_processed)
        st.write("Predicted GMV:", prediction[0]**2)  # Check if squaring is necessary
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
