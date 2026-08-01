import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
model = joblib.load(model_path)

# Streamlit App Title
st.title("Wellness Tourism Package Purchase Predictor")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Input fields for customer details
st.header("Customer Information")

age = st.slider("Age", 18, 90, 30)
type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lance', 'Student'])
gender = st.selectbox("Gender", ['Male', 'Female'])
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
number_of_trips = st.number_input("Number of Trips Annually", 0, 50, 5)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'President', 'Director'])
monthly_income = st.number_input("Monthly Income", 10000, 500000, 50000)

st.header("Interaction Details")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba > 0.5).astype(int)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The customer is LIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")
    else:
        st.warning(f"The customer is UNLIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")
    st.write(f"*Note: Probability > 0.5 is classified as 'Likely to purchase'.*")
