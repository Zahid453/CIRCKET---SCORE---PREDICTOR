import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

# Load the pre-trained pipeline
try:
    pipe = pickle.load(open('Final_Pickle.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")

# List of teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 
          'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 
          'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 
          'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 
          'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 
          'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 
          'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Title of the app
st.title('Cricket Score Predictor')

# Input fields
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs done (works for overs > 5)', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

last_five = st.number_input('Runs scored in last 5 overs', min_value=0, step=1)

# When the user clicks the 'Predict Score' button
if st.button('Predict Score'):
    try:
        # Handle case where overs is 0 to avoid division by zero
        if overs == 0:
            st.error("Overs cannot be 0 for prediction.")
        else:
            # Calculate additional features
            balls_left = 120 - (overs * 6)
            wickets_left = 10 - wickets
            crr = current_score / overs

            # Prepare input data in a DataFrame
            input_data = {
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [current_score],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'crr': [crr],
                'last_five': [last_five]
            }

            input_df = pd.DataFrame(input_data)

            # Ensure input_df has the correct structure and data types for the pipeline
            result = pipe.predict(input_df)

            # Display the predicted score
            st.header(f"Predicted Score - {int(result[0])}")
    
    except Exception as e:
        # Display an error message if there's an issue with the prediction
        st.error(f"An error occurred during prediction: {e}")
