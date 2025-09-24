import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Lifestyle & Heart Risk Predictor", layout="wide")
st.title("🩺 Lifestyle & Heart Risk Predictor + CSV Insights")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", [
    "🏃 Manual Lifestyle Input",
    "📊 CSV Upload Comparisons"
])

# -------------------------------
# PAGE 1: Manual Lifestyle Input
# -------------------------------
if page == "🏃 Manual Lifestyle Input":
    st.header("Enter your lifestyle & health data for Heart Risk Prediction")

    # Numeric inputs
    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)

    # Lifestyle categorical inputs
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    # Convert categorical to numeric
    sex = 0 if sex == "Male" else 1
    smoke_map = {"Never": 0, "Used to": 1, "Occasionally": 2, "Regularly": 3}
    alco_map = {"Never": 0, "Occasionally": 1, "Regularly": 2}
    active_map = {"Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3}
    smoke = smoke_map[smoke]
    alco = alco_map[alco]
    active = active_map[active]

    if st.button("Predict Heart Risk"):
        input_data = pd.DataFrame({
            'age':[age],
            'sex':[sex],
            'resting_bp':[resting_bp],
            'cholesterol':[cholesterol],
            'max_hr':[max_hr],
            'steps_per_day':[steps_per_day],
            'sedentary_hours':[sedentary_hours],
            'smoke':[smoke],
            'alco':[alco],
            'active':[active]
        })

        # Dummy model for demonstration
        X_demo = input_data.copy()
        y_demo = [1]  # assume high risk
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_demo, y_demo)
        pred = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write("High Risk ⚠️" if pred==1 else "Low Risk ✅")

        # Tips
        st.subheader("Lifestyle Tips")
        if pred == 1:
            st.markdown("""
            - Increase physical activity: aim for at least 10k steps/day  
            - Reduce sedentary hours: take breaks every hour  
            - Eat a balanced diet low in saturated fat and sugar  
            - Avoid smoking & limit alcohol intake  
            - Regular check-ups with your doctor
            """)
        else:
            st.markdown("Keep maintaining your healthy lifestyle! 💪")

# -------------------------------
# PAGE 2: CSV Upload Comparisons
# -------------------------------
elif page == "📊 CSV Upload Comparisons":
    st.header("Upload CSV files for batch heart risk predictions & comparisons")
    uploaded_files = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        all_dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            df['Source File'] = file.name
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        st.subheader("Combined Data Preview")
        st.dataframe(combined_df.head())

        # Take only numeric columns for dummy prediction
        numeric_cols = combined_df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for prediction.")
        else:
            X = combined_df[numeric_cols]
            y_dummy = np.random.randint(0,2,len(combined_df))  # dummy target
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y_dummy)
            combined_df['Prediction'] = model.predict(X)

            st.subheader("Prediction Distribution Across Files")
            fig = px.bar(
                combined_df['Prediction'].value_counts().rename({0:'Low Risk', 1:'High Risk'}),
                labels={'index':'Risk Category','value':'Count'},
                title="Overall Heart Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("File-wise Summary")
            summary = combined_df.groupby('Source File')['Prediction'].value_counts().unstack(fill_value=0)
            st.dataframe(summary)

            # Download button
            st.download_button(
                label="Download Predictions CSV",
                data=combined_df.to_csv(index=False),
                file_name="Predicted_Heart_Risk.csv",
                mime="text/csv"
            )




