import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from requests.exceptions import HTTPError
import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie

from sklearn.preprocessing import MinMaxScaler



# Page Layout
st.set_page_config(
    page_title="PCOS Prediction",
    layout="wide",
    page_icon="📈"
)

# Title
st.title('PCOS Prediction Dashboard')

# Four columns for inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("Left Ovarian Follicle Count")
    left_follicle_value = st.number_input("Enter left follicle count", value=0.0, step=0.01, format="%.4f")

with col2:
    st.write("Right Ovarian Follicle Count")
    right_follicle_value = st.number_input("Enter right follicle count", value=0.0, step=0.01, format="%.4f")

with col3:
    st.write("AMH (ng/ml)")
    amh_value = st.number_input("Enter AMH value", value=0.0, step=0.01, format="%.4f")

with col4:
    st.write("Weight (kg)")
    weight_value = st.number_input("Enter weight", value=0.0, step=0.01, format="%.4f")



col5, col6  = st.columns(2)
with col5:
    st.write("Hair-Growth Value")
    hair_growth_value = st.selectbox("Do you have excessive hair growth ?", ["Yes", "No"])

with col6:
    st.write("Skin-Colour")
    skin_dark_bol_value = st.selectbox("Skin-Darkening ?", ["Yes", "No"])


col7, col8  = st.columns(2)
with col7:
    st.write("Cycle Hour")
    menstrual_cycle_bol_value = st.selectbox("Menstrual Cycle?", ["4 [ Regular Cycles ]", "2 [ Irregular Cycles ]"])

with col8:
    st.write("Fast Food")
    fast_food_bol_value = st.selectbox("Fast Food Eater?", ["Yes", "No"])



col9, col10  = st.columns(2)
with col9:
    st.write("Pimples")
    pimples__bol_value = st.selectbox("Have Pimples?", ["Yes", "No"])

with col10:
    st.write("Weight Gain Data")
    weight_gain_bol_value = st.selectbox("Weight Gain recently?", ["Yes", "No"])




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


X = data[['Follicle No. (R)', 'Follicle No. (L)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
           'Weight gain(Y/N)', 'Cycle(R/I)', 'Fast food (Y/N)', 'AMH(ng/mL)',
           'Pimples(Y/N)', 'Weight (Kg)']]
y = data['PCOS (Y/N)']

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



model = tf.keras.models.load_model('mlp_model.h5', compile=False)

# Step 13: Make predictions on user input
user_pred_prob = model.predict(user_input_scaled)
user_pred = (user_pred_prob > 0.5).astype(int)

# Step 14: Display prediction result
print(f"Prediction (PCOS 1=Yes, 0=No): {user_pred[0][0]}")