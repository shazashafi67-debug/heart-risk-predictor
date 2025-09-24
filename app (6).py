import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Lifestyle & Heart Risk Predictor", layout="wide")
st.title("🩺 Lifestyle & Heart Risk Predictor")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", [
    "🏃 Manual Lifestyle Input",
    "📊 CSV Upload Predictions",
    "🎯 Goal Setting & Progress Tracking"
])


# PAGE 1: Manual Lifestyle Input

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
# PAGE 2: CSV Upload Predictions
# -------------------------------
elif page == "📊 CSV Upload Predictions":
    st.header("Upload CSV files for batch heart risk predictions")
    uploaded_files = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Preview: {file.name}")
            df = pd.read_csv(file)
            st.dataframe(df.head())

            try:
                # Take only numeric columns for prediction
                X = df.select_dtypes(include=np.number)
                y_dummy = np.random.randint(0,2,len(df))  # dummy target
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y_dummy)
                df['Prediction'] = model.predict(X)

                st.subheader("Prediction Distribution")
                fig = px.bar(
                    x=['Low Risk','High Risk'],
                    y=[(df['Prediction']==0).sum(), (df['Prediction']==1).sum()],
                    labels={'x':'Risk Category','y':'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Could not predict for `{file.name}`: {e}")

# -------------------------------
# PAGE 3: Goal Setting & Progress Tracking
# -------------------------------
elif page == "🎯 Goal Setting & Progress Tracking":
    st.header("Set your health goals and track your progress!")

    # User inputs for goals
    steps_goal = st.number_input("Daily Steps Goal", 1000, 50000, 10000)
    exercise_goal = st.number_input("Daily Exercise Minutes Goal", 0, 180, 30)
    water_goal = st.number_input("Daily Water Intake (ml)", 500, 5000, 2000)

    # User logs
    st.subheader("Log Today's Progress")
    steps_done = st.number_input("Steps Done Today", 0, 50000, 0)
    exercise_done = st.number_input("Exercise Minutes Today", 0, 180, 0)
    water_done = st.number_input("Water Intake Today (ml)", 0, 5000, 0)

    # Calculate progress %
    steps_pct = min(steps_done / steps_goal * 100, 100)
    exercise_pct = min(exercise_done / exercise_goal * 100, 100)
    water_pct = min(water_done / water_goal * 100, 100)

    st.subheader("Progress Overview")
    st.write(f"Steps: {steps_done}/{steps_goal} ({steps_pct:.1f}%)")
    st.write(f"Exercise: {exercise_done}/{exercise_goal} min ({exercise_pct:.1f}%)")
    st.write(f"Water Intake: {water_done}/{water_goal} ml ({water_pct:.1f}%)")

    progress_df = pd.DataFrame({
        "Metric":["Steps","Exercise","Water Intake"],
        "Progress %":[steps_pct, exercise_pct, water_pct]
    })

    fig = px.bar(progress_df, x="Metric", y="Progress %", text="Progress %", range_y=[0,100])
    st.plotly_chart(fig, use_container_width=True)





