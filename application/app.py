import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data



import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf

# Use the recommended import path for LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split




# Page Layout
st.set_page_config(
    page_title="PCOS Prediction",
    layout="wide",
    page_icon="⚕️"
)


with st.sidebar:
    st.title("PCOS Prediction using Multi-Layer Perceptron")
    st.subheader("PCOS Identifier")
    st.markdown(
        """The objective of creating a PCOS Prediction project using a Multi-Layer Perceptron (MLP) in Streamlit is to develop an accessible, user-friendly application that aids in the early detection of Polycystic Ovary Syndrome (PCOS). By leveraging machine learning techniques, this project aims to analyze user-inputted medical data—such as ovarian follicle counts, AMH levels, and lifestyle factors—to predict the likelihood of PCOS. The application will provide users with instant feedback on their health status, empowering them to make informed decisions. Ultimately, this project seeks to raise awareness about PCOS, encourage proactive health management, and enhance patient outcomes through technology."""
    )

    st.success("Deployes")

# Title
st.title('Dashboard')

data = pd.read_csv("C:/Users/Y.Tushaar/PCOS/training/cleaned_data.csv")
st.dataframe(data)

# Four columns for inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("Left Ovarian follicle Count")
    follicle_r = st.number_input(
    "Enter left follicle count", 
    value=0,  
    min_value=0,  
    max_value=30,  
    step=1,  
    format="%d"  
)

with col2:
    st.write("Right Ovarian Follicle Count")
    follicle_l = st.number_input(
        "Enter right follicle count", 
        min_value=0,  
        max_value=30,  
        value=0, 
        step=1,  
        format="%d" 
    )


with col3:
    st.write("AMH (ng/ml)")
    amh = st.number_input(
        "Enter AMH value", 
        min_value=1.0,  
        max_value=20.0,  
        value=1.0,  
        step=0.01,  
        format="%.4f" 
    )

with col4:
    st.write("Weight (kg)")
    weight_kg = st.number_input("Enter weight", min_value=40.0, max_value=170.0, step=0.1, format="%.1f")



col5, col6  = st.columns(2)
with col5:
    st.write("Hair-Growth Value")
    hair_growth = st.selectbox("Do you have excessive hair growth ?", ["Yes", "No"])

with col6:
    st.write("Skin-Colour")
    skin_darkening = st.selectbox("Skin-Darkening ?", ["Yes", "No"])


col7, col8  = st.columns(2)
with col7:
    st.write("Cycle Hour")
    cycle = st.selectbox("Menstrual Cycle?", ["4 [ Regular Cycles ]", "2 [ Irregular Cycles ]"])

with col8:
    st.write("Fast Food")
    fast_food = st.selectbox("Fast Food Eater?", ["Yes", "No"])



col9, col10  = st.columns(2)
with col9:
    st.write("Pimples")
    pimples = st.selectbox("Have Pimples?", ["Yes", "No"])

with col10:
    st.write("Weight Gain Data")
    weight_gain = st.selectbox("Weight Gain recently?", ["Yes", "No"])




## Download model
model_file_path='mlp_model.h5'

with open(model_file_path, 'rb' ) as f:
    model_data=f.read()

st.download_button(
    label="Download Model (.h5 format)",
    data=model_data,
    file_name="model.h5",
    mime="application/octet-stream"
)



target_variable = 'PCOS (Y/N)'

X = data[['Follicle No. (R)', 'Follicle No. (L)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
           'Weight gain(Y/N)', 'Cycle(R/I)', 'Fast food (Y/N)', 'AMH(ng/mL)',
           'Pimples(Y/N)', 'Weight (Kg)']]
y = data[target_variable]

# Step 2: Drop rows with any missing values in X
X = X.dropna()

# Step 3: Also drop the corresponding rows in the target variable to keep the alignment
y = y.loc[X.index]

# Step 4: Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['hair growth(Y/N)', 'Skin darkening (Y/N)', 'Weight gain(Y/N)',
                               'Cycle(R/I)', 'Fast food (Y/N)', 'Pimples(Y/N)'], drop_first=True)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Align columns in training and testing sets
missing_cols_train = set(X_test.columns) - set(X_train.columns)
for c in missing_cols_train:
    X_train[c] = 0

missing_cols_test = set(X_train.columns) - set(X_test.columns)
for c in missing_cols_test:
    X_test[c] = 0
X_test = X_test[X_train.columns]

# Step 6: Normalize the feature data and store column names
scaler = StandardScaler()
X_train_columns = X_train.columns  # Store the column names before scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Build the neural network model with L2 regularization and Dropout
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Step 8: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Implement EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 10: Fit the model to the training data
history = model.fit(X_train, y_train, epochs=500, batch_size=20, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Step 11: Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

def get_user_input(follicle_r, follicle_l, hair_growth, skin_darkening, weight_gain, cycle, fast_food, amh, pimples, weight_kg):

    hair_growth_numeric = 1 if hair_growth == 'Yes' else 0
    skin_darkening_numeric = 1 if skin_darkening == 'Yes' else 0
    weight_gain_numeric = 1 if weight_gain == 'Yes' else 0
    fast_food_numeric = 1 if fast_food == 'Yes' else 0
    pimples_numeric = 1 if pimples == 'Yes' else 0

    user_data = pd.DataFrame([[follicle_r, follicle_l, hair_growth_numeric, skin_darkening_numeric,
                                weight_gain_numeric, cycle, fast_food_numeric, amh, pimples_numeric, weight_kg]],
                              columns=['Follicle No. (R)', 'Follicle No. (L)', 'hair growth(Y/N)', 
                                       'Skin darkening (Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)', 
                                       'Fast food (Y/N)', 'AMH(ng/mL)', 'Pimples(Y/N)', 'Weight (Kg)'])

    # Perform one-hot encoding to match the training data structure
    user_data_encoded = pd.get_dummies(user_data, columns=['Cycle(R/I)'], drop_first=True)

    # Align columns with the stored column names from the training data
    for col in X_train_columns:
        if col not in user_data_encoded.columns:
            user_data_encoded[col] = 0

    # Ensure the order of columns matches
    user_data_encoded = user_data_encoded[X_train_columns]

    return user_data_encoded

user_input = get_user_input(follicle_r, follicle_l, hair_growth, skin_darkening, weight_gain, cycle, fast_food, amh, pimples, weight_kg)
user_input_scaled = scaler.transform(user_input)


model = tf.keras.models.load_model('mlp_model.h5', compile=False)

# Step 13: Make predictions on user input
user_pred_prob = model.predict(user_input_scaled)
user_pred = (user_pred_prob > 0.5).astype(int)

# Step 14: Display prediction result
if (user_pred[0][0]) == 0:
    st.subheader(f"Yes, You are have PCOS.")
elif (user_pred[0][0]) == 1:
    st.subheader(f"No, You don't have PCOS.")
else:
    st.alert(f"Error: 404")
