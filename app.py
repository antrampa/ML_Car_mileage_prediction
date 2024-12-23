import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Car Mileage Predictor",
    page_icon="🚗",
    layout="centered"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚗 Car Mileage Predictor")
st.markdown("""
This app predicts the mileage (MPG) of a car based on its horsepower and weight.
Please enter the required information below to get a prediction.
""")

# Create input form
with st.form("prediction_form"):
    st.subheader("Enter Car Details")
    
    # Input fields
    horsepower = st.number_input(
        "Horsepower (HP)",
        min_value=0,
        max_value=1000,
        value=120,
        help="Enter the horsepower of the car"
    )
    
    weight = st.number_input(
        "Weight (lbs)",
        min_value=0,
        max_value=10000,
        value=2000,
        help="Enter the weight of the car in pounds"
    )
    
    # Submit button
    submit = st.form_submit_button("Predict Mileage")

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file not found. Please ensure 'model.pkl' is in the same directory as the app.")
    st.stop()

# Make prediction when form is submitted
if submit:
    try:
        # Create input data frame
        input_data = pd.DataFrame({
            'HP': [horsepower],
            'Wt': [weight]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.success("Prediction Complete!")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col2:
            st.metric(
                label="Predicted Mileage",
                value=f"{prediction:.2f} MPG"
            )
        
        # Add some context
        st.info("""
        💡 Interpretation:
        - Higher horsepower typically results in lower MPG
        - Heavier cars generally have lower MPG
        - This prediction is based on historical car data
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add additional information
st.markdown("---")
st.markdown("""
### About the Model
This prediction model uses Linear Regression trained on historical car data. 
The model takes into account:
- Horsepower (HP)
- Weight (Wt)

to predict the Miles Per Gallon (MPG) a car might achieve.
""")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Made with ❤️ for Car Enthusiasts")