import streamlit as st
import joblib
import numpy as np

# Load the scaler and trained model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.set_page_config(page_title="Employee Performance Predictor", layout="centered")
st.title("🧑‍💼 Employee Performance Predictor")
st.markdown("Predict employee performance based on key HR indicators.")

st.divider()

with st.form("prediction_form"):
    st.subheader("🔍 Enter Employee Details")

    years = st.number_input("Years at Company", min_value=0, max_value=50, value=2, step=1)
    salary = st.number_input("Monthly Salary (₹)", min_value=1000, max_value=200000, value=30000, step=500)
    overtime = st.number_input("Overtime Hours per Month", min_value=0, max_value=200, value=10, step=1)
    promotions = st.number_input("Number of Promotions", min_value=0, max_value=10, value=1, step=1)
    satisfaction = st.slider("Employee Satisfaction Score", 0.0, 1.0, value=0.6, step=0.01)

    submitted = st.form_submit_button("🎯 Predict Performance")

# Only run prediction if form was submitted
if submitted:
    # Prepare input data
    input_data = np.array([[years, salary, overtime, promotions, satisfaction]])
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Show result
    st.markdown("---")
    st.subheader("📈 Prediction Result")
    st.success(f"🏆 Predicted Performance Score: **{prediction[0]}**")
    st.markdown("This score is based on the employee's experience, salary, workload, and satisfaction.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
