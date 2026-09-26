import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model from the Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ASNaik/tourism-wellness-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the newly
introduced **Wellness Tourism Package**, based on their profile and their
past interaction with the sales team. Fill in the details below and click
**Predict** to see the result.
""")

st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=25, value=3)
passport = st.selectbox("Holds Passport?", ["Yes", "No"])
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
num_children_visiting = st.number_input("Number of Children (below 5 yrs) Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0, step=500.0)

st.header("Sales Interaction Details")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0)

# Assemble the inputs into a single-row DataFrame matching the training columns
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The model predicts: **Likely to Purchase** the Wellness Tourism Package")
    else:
        st.info("The model predicts: **Not Likely to Purchase** the Wellness Tourism Package")
