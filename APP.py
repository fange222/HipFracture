import streamlit as st
import joblib
import pandas as pd

# Load the AdaBoost model
model = joblib.load('adaboost.pkl')

# 定义连续变量和分类变量名称
feature_names_continuous = ["BUN", "HCT", "AST", "FIB", "Ca"]
feature_names_categorical = ["Age", "ASA", "Dementia", "Stroke", "AS", "Bloodtransfusion"]

# Streamlit user interface
st.title("Survival Prediction")

# Continuous variable inputs
st.header("Continuous Variables")
bun = st.number_input("BUN (Blood Urea Nitrogen):", min_value=0.0, max_value=50.0, value=20.0)
hct = st.number_input("HCT (Hematocrit):", min_value=0.0, max_value=50.0, value=30.0)
ast = st.number_input("AST (Aspartate Aminotransferase):", min_value=0.0, max_value=200.0, value=40.0)
fib = st.number_input("FIB (Fibrinogen):", min_value=0.0, max_value=10.0, value=3.0)
ca = st.number_input("Ca (Calcium):", min_value=0.0, max_value=5.0, value=2.5)

# Categorical variable inputs
st.header("Categorical Variables")
age = st.selectbox("Age:", options=[1, 2], format_func=lambda x: "<80" if x == 1 else "≥80")
asa = st.selectbox("ASA Grading:", options=[1, 2, 3], format_func=lambda x: "II" if x == 1 else ("III" if x == 2 else "IV"))
dementia = st.selectbox("Dementia:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
stroke = st.selectbox("Stroke:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
as_ = st.selectbox("AS (Arteriosclerosis):", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
bloodtransfusion = st.selectbox("Intraoperative Blood Transfusion:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Combine input values
feature_values_continuous = [bun, hct, ast, fib, ca]
feature_values_categorical = [age, asa, dementia, stroke, as_, bloodtransfusion]
feature_values = feature_values_continuous + feature_values_categorical

# Create a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names_continuous + feature_names_categorical)

# 手动设置列名以匹配模型期望的特征名称
features_df.columns = model.feature_names_in_

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"According to our model, the patient is predicted to survive. "
            f"The model predicts a {probability:.1f}% probability of survival. "
            "It is advised to maintain regular check-ups."
        )
    else:
        advice = (
            f"According to our model, the patient is predicted not to survive. "
            f"The model predicts a {probability:.1f}% probability of death. "
            "It is strongly recommended to seek further medical evaluation."
        )

    st.write(advice)

