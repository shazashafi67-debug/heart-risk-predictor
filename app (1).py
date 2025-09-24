import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Heart Risk Predictor CSV", layout="wide")
st.title("💓 Heart Risk Predictor")

# -------------------- Tabs --------------------
tabs = st.tabs(["Predict Online", "Upload CSVs", "Model Accuracy"])

# -------------------- TAB 1: Online Input --------------------
with tabs[0]:
    st.subheader("Enter Your Details:")

    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    sleep_hours = st.number_input("Sleep Duration (hours/night)", 3, 12, 7)
    physical_activity = st.number_input("Minutes of Activity per Day", 0, 300, 30)

    sex = st.selectbox("Sex", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["Non-smoker", "Former Smoker", "Current Smoker"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    diet = st.selectbox("Diet Quality", ["Poor", "Average", "Healthy"])
    stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

    def calculate_risk(row):
        score = 0
        score += 1 if row['age'] > 50 else 0
        score += 1 if row['resting_bp'] > 130 else 0
        score += 1 if row['cholesterol'] > 240 else 0
        score += 1 if row['physical_activity'] < 30 else 0
        score += 1 if row['smoking'] != "Non-smoker" else 0
        score += 1 if row['alcohol'] == "High" else 0
        score += 1 if row['diet'] == "Poor" else 0
        score += 1 if row['stress'] == "High" else 0
        return score

    input_data = pd.DataFrame([{
        'age': age,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'physical_activity': physical_activity,
        'sex': sex,
        'smoking': smoking,
        'alcohol': alcohol,
        'diet': diet,
        'sleep_hours': sleep_hours,
        'stress': stress
    }])

    if st.button("Predict Risk"):
        input_data['Risk_Score'] = input_data.apply(calculate_risk, axis=1)
        input_data['Risk_Level'] = input_data['Risk_Score'].apply(lambda x: "High Risk ⚠️" if x>=4 else "Low Risk ✅")
        st.subheader("Prediction:")
        st.write(input_data[['Risk_Score','Risk_Level']])

        if input_data['Risk_Level'][0] == "High Risk ⚠️":
            st.markdown("### 🩺 Tips for High Risk:")
            st.markdown("- Eat heart-healthy foods and reduce saturated fat.")
            st.markdown("- Exercise at least 30 min/day.")
            st.markdown("- Monitor blood pressure and cholesterol regularly.")
            st.markdown("- Avoid smoking and limit alcohol consumption.")
            st.markdown("- Reduce stress and get adequate sleep.")

# -------------------- TAB 2: CSV Uploads --------------------
with tabs[1]:
    st.subheader("Upload Multiple CSV Files")
    uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        combined_df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)

            expected_columns = ['age','resting_bp','cholesterol','physical_activity','sex','smoking','alcohol','diet','sleep_hours','stress']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0 if col in ['age','resting_bp','cholesterol','physical_activity','sleep_hours'] else "Unknown"

            df['Risk_Score'] = df.apply(calculate_risk, axis=1)
            df['Risk_Level'] = df['Risk_Score'].apply(lambda x: "High Risk ⚠️" if x>=4 else "Low Risk ✅")
            df['Tips'] = df['Risk_Level'].apply(lambda x: "- Heart-healthy diet, exercise, monitor BP & cholesterol, avoid smoking/alcohol, reduce stress." if x=="High Risk ⚠️" else "")

            combined_df = pd.concat([combined_df, df], ignore_index=True)

        st.subheader("Combined Predictions")
        st.dataframe(combined_df)

        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Predictions", csv, "Predicted_Heart_Risk.csv", "text/csv")

# -------------------- TAB 3: Model Accuracy --------------------
with tabs[2]:
    st.subheader("Model Accuracy (Rule-based Example)")
    data = {'Heart Risk Model': 80, 'Cardio Fitness Model': 75, 'Activity Level Model': 70}
    df_acc = pd.DataFrame(list(data.items()), columns=['Model','Accuracy'])
    fig = px.bar(df_acc, x='Model', y='Accuracy', color='Accuracy', text='Accuracy')
    st.plotly_chart(fig)

