import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Student Mental Health",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Overthinking & Mental Health")
st.write("An educational machine-learning project exploring student survey data.")

st.warning(
    "Educational demo only. This tool does not diagnose mental-health "
    "conditions or replace advice from a qualified professional."
)

DATA_FILE = Path("Student_Mental_Health_Cleaned.csv")

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Check that Student_Mental_Health_Cleaned.csv "
        "is uploaded to the GitHub repository."
    )
    st.stop()

df = pd.read_csv(DATA_FILE)

gender_col = "Choose your gender"
age_col = "Age"
course_col = "What is your course?"
year_col = "Your current year of Study"
cgpa_col = "What is your CGPA?"
depression_col = "Do you have Depression?"
anxiety_col = "Do you have Anxiety?"
panic_col = "Do you have Panic attack?"

required = [
    gender_col, age_col, course_col, year_col, cgpa_col,
    depression_col, anxiety_col, panic_col
]

if not all(col in df.columns for col in required):
    st.error("The dataset columns do not match the expected format.")
    st.stop()

def calculate_level(row):
    count = sum(
        row[col] == "Yes"
        for col in [depression_col, anxiety_col, panic_col]
    )
    return [
        "No Overthinking",
        "Low Overthinking",
        "Medium Overthinking",
        "High Overthinking"
    ][count]

df = df.dropna(subset=required).copy()
df["Overthinking_Level"] = df.apply(calculate_level, axis=1)

feature_cols = [gender_col, age_col, course_col, year_col, cgpa_col]
X = df[feature_cols].copy()
y = df["Overthinking_Level"]

encoders = {}
for col in feature_cols:
    if X[col].dtype == "object":
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        encoders[col] = encoder

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("There are not enough records in each category to train this demo.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Survey records", len(df))
c2.metric("Input features", len(feature_cols))
c3.metric("Test accuracy", f"{test_accuracy:.1%}")

st.caption(
    "Accuracy is measured on a small held-out portion of this dataset. "
    "It is not evidence of clinical accuracy or reliable diagnosis."
)

st.subheader("🔎 Explore the survey")
st.bar_chart(df["Overthinking_Level"].value_counts())

with st.expander("View sample survey records"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.subheader("🧠 Demo prediction")
st.write("Enter academic and demographic details to explore the model output.")

with st.form("prediction_form"):
    gender = st.selectbox(
        "Gender", sorted(df[gender_col].astype(str).unique())
    )
    age = st.number_input("Age", min_value=16, max_value=100, value=20)
    course = st.selectbox(
        "Course", sorted(df[course_col].astype(str).unique())
    )
    year = st.selectbox(
        "Year of study", sorted(df[year_col].astype(str).unique())
    )
    cgpa = st.selectbox(
        "CGPA range", sorted(df[cgpa_col].astype(str).unique())
    )
    submitted = st.form_submit_button("Explore prediction")

if submitted:
    input_data = pd.DataFrame([{
        gender_col: gender,
        age_col: age,
        course_col: course,
        year_col: year,
        cgpa_col: cgpa
    }])

    for col, encoder in encoders.items():
        value = str(input_data.loc[0, col])
        if value not in encoder.classes_:
            st.error(f"Unsupported value for {col}.")
            st.stop()
        input_data[col] = encoder.transform([value])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    st.info(f"Model output: **{predicted_label}**")
    st.write("Model scores for each category (not clinical probabilities):")

    scores = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Model score": probabilities
    }).sort_values("Model score", ascending=False)

    st.bar_chart(scores.set_index("Category"))

st.caption(
    "This project uses a survey of approximately 101 students. "
    "The output is experimental and must not be used for medical decisions."
)
