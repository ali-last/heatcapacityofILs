import streamlit as st
import pandas as pd
import joblib

# Load the saved XGBoost model
model = joblib.load('xgboost_model.pkl')

import base64

# Specify the local image file name
image_path = "background.png"  # Change this to your image filename

# Function to load the image and convert it to base64
def load_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load the image
image_base64 = load_image(image_path)

# Inject custom CSS to apply the background image
page_bg_img = f'''
<style>
.stApp {{
    background-image: url(data:image/jpeg;base64,{image_base64});
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    height: 100vh;  /* Cover the whole height of the viewport */
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app title
st.title('Prediction of molar heat capacity of pure liquid-phase ionic liquids using XGBoost model')

# Instructions
st.write("""
### Input the features below to get the predicted heat capacity value.
""")

# Create input fields for the features using their exact names
F1 = st.number_input('T (K):', min_value=183.45, max_value=554.94, value=183.45)
F2 = st.number_input('  -CH3:', min_value=0, max_value=12, value=2)
F3 = st.number_input(' -CH2-', min_value=0, max_value=44, value=3)
F4 = st.number_input('\>CH-', min_value=0, max_value=3, value=0)
F5 = st.number_input('\>C<', min_value=0, max_value=6, value=0)
F6 = st.number_input(' =CH2', min_value=0, max_value=2, value=0)
F7 = st.number_input(' =CH-', min_value=0, max_value=2, value=0)
F8 = st.number_input(' -OH', min_value=0, max_value=3, value=0)
F9 = st.number_input(' -O-', min_value=0, max_value=12, value=0)
F10 = st.number_input('\>C=O', min_value=0, max_value=1, value=0)
F11 = st.number_input(' -COOH', min_value=0, max_value=1, value=0)
F12 = st.number_input(' -COO-', min_value=0, max_value=4, value=0)
F13 = st.number_input('HCOO-', min_value=0, max_value=1, value=0)
F14 = st.number_input(' =O(other)', min_value=0, max_value=1, value=0)
F15 = st.number_input(' -NH2', min_value=0, max_value=2, value=0)
F16 = st.number_input(' -NH3', min_value=0, max_value=1, value=0)
F17 = st.number_input(' -NH-', min_value=0, max_value=1, value=0)
F18 = st.number_input('\>N-', min_value=0, max_value=3, value=0)
F19 = st.number_input(' =N-', min_value=0, max_value=1, value=0)
F20 = st.number_input(' -CN', min_value=0, max_value=4, value=0)
F21 = st.number_input(' -NO2', min_value=0, max_value=1, value=0)
F22 = st.number_input(' -F', min_value=0, max_value=18, value=4)
F23 = st.number_input(' -Cl', min_value=0, max_value=1, value=0)
F24 = st.number_input(' -Br', min_value=0, max_value=2, value=0)
F25 = st.number_input(' -I', min_value=0, max_value=1, value=0)
F26 = st.number_input(' -P', min_value=0, max_value=2, value=0)
F27 = st.number_input(' -B', min_value=0, max_value=1, value=1)
F28 = st.number_input(' -S-', min_value=0, max_value=1, value=0)
F29 = st.number_input(' -SO2', min_value=0, max_value=2, value=0)
F30 = st.number_input(' -CH2- (ring)', min_value=0, max_value=6, value=0)
F31 = st.number_input('\>CH- (ring)', min_value=0, max_value=3, value=0)
F32 = st.number_input(' =CH- (ring)', min_value=0, max_value=8, value=3)
F33 = st.number_input(' =C< (ring)', min_value=0, max_value=4, value=0)
F34 = st.number_input(' -NH- (ring)', min_value=0, max_value=2, value=0)
F35 = st.number_input(' \>N- (ring)', min_value=0, max_value=2, value=1)
F36 = st.number_input(' =N- (ring)', min_value=0, max_value=2, value=1)


# Convert the inputs into a DataFrame using exact feature names
input_data = pd.DataFrame({
    'T (K)': [F1],
    '  -CH3': [F2],
    ' -CH2-': [F3],
    '>CH-': [F4],
    '>C': [F5],
    ' =CH2': [F6],
    ' =CH-': [F7],
    ' -OH': [F8],
    ' -O-': [F9],
    '>C=O': [F10],
    ' -COOH': [F11],
    ' -COO-': [F12],
    'HCOO-': [F13],
    ' =O(other)': [F14],
    ' -NH2': [F15],
    ' -NH3': [F16],
    ' -NH-': [F17],
    '>N-': [F18],
    ' =N-': [F19],
    ' -CN': [F20],
    ' -NO2': [F21],
    ' -F': [F22],
    ' -Cl': [F23],
    ' -Br': [F24],
    ' -I': [F25],
    ' -P': [F26],
    ' -B': [F27],
    ' -S-': [F28],
    ' -SO2': [F29],
    ' -CH2- (ring)': [F30],
    '>CH- (ring)': [F31],
    ' =CH- (ring)': [F32],
    ' =C (ring)': [F33],
    ' -NH- (ring)': [F34],
    ' >N- (ring)': [F35],
    ' =N- (ring)': [F36]


})

# When the user clicks the "Predict" button, make a prediction
if st.button('Run the prediction'):
    prediction = model.predict(input_data)
    st.write(f'Predicted heat capacity for the input IL is: {prediction[0]:.4f}')
