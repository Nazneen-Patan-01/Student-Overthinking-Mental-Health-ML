# 🧠 Student Overthinking & Mental Health — EDA + ML Project

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-2.4.0-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end Data Science project analyzing **student mental health** — covering data cleaning, exploratory data analysis (EDA), and machine learning classification models (Logistic Regression & Random Forest) to predict Depression, Anxiety, and Panic Attacks.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [EDA Highlights](#-eda-highlights)
- [Machine Learning Models](#-machine-learning-models)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Project Overview

Mental health among university students is a growing concern worldwide. This project uses a real-world survey dataset to:

1. **Clean & preprocess** the raw student mental health data
2. **Explore patterns** through comprehensive EDA (univariate, bivariate, and multivariate analysis)
3. **Build ML classification models** to predict whether a student suffers from **Depression**, **Anxiety**, or **Panic Attacks**
4. **Analyze overthinking levels** based on combined mental health indicators

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | University student mental health survey |
| **Records** | 101 students |
| **Original Features** | 11 columns |
| **Cleaned Features** | 8 columns |
| **File** | `Student Mental health.csv` |

### Original Columns (11)

| Column | Type | Description |
|---|---|---|
| Timestamp | Object | Survey response timestamp *(removed)* |
| Choose your gender | Object | Student gender |
| Age | Float | Student age |
| What is your course? | Object | Academic course/program |
| Your current year of Study | Object | Year 1–4 |
| What is your CGPA? | Object | CGPA range (e.g., 3.00–3.49) |
| Marital status | Object | Marital status *(removed)* |
| Do you have Depression? | Object | Yes/No |
| Do you have Anxiety? | Object | Yes/No |
| Do you have Panic attack? | Object | Yes/No |
| Did you seek any specialist for a treatment? | Object | *(removed)* |

### Cleaned Dataset (8 columns)
After removing irrelevant columns (`Timestamp`, `Marital status`, `Did you seek any specialist for a treatment?`):

```
Gender | Age | Course | Year of Study | CGPA | Depression | Anxiety | Panic Attack
```

---

## 📁 Project Structure

```
mental-health/
│
├── 📓 Clean_Mental_Health_Data.ipynb          # Data cleaning notebook
├── 📓 EDA_Analysis.ipynb                      # Full exploratory data analysis
├── 📓 Logistic_Regression.ipynb               # Logistic Regression model
├── 📓 Random_Forest.ipynb                     # Random Forest model
├── 📓 Overthinking_Level_Prediction.ipynb     # Overthinking level analysis
│
├── 📊 Student Mental health.csv               # Original raw dataset
├── 📊 Student_Mental_Health_Cleaned.csv       # Cleaned dataset
│
├── 🖼️ EDA_Age_Analysis.png
├── 🖼️ EDA_BoxPlots.png
├── 🖼️ EDA_Categorical.png
├── 🖼️ EDA_Course_CGPA.png
├── 🖼️ EDA_Distributions.png
├── 🖼️ EDA_Gender_MentalHealth.png
├── 🖼️ EDA_Heatmap.png
├── 🖼️ EDA_Mental_Health.png
├── 🖼️ Logistic_Regression_Analysis.png
├── 🖼️ Random_Forest_Analysis.png
└── 🖼️ Overthinking_Level_Analysis.png
```

---

## 🔄 Workflow

```
Raw CSV Data
     │
     ▼
1️⃣  Data Cleaning
     │  → Remove irrelevant columns (Timestamp, Marital Status, Treatment)
     │  → Standardize Year of Study formatting
     │  → Handle missing Age values (1 missing)
     ▼
2️⃣  Exploratory Data Analysis (EDA)
     │  → Univariate analysis (distributions, age, gender)
     │  → Bivariate analysis (Gender vs Mental Health, Course vs CGPA)
     │  → Correlation heatmap
     │  → Categorical breakdowns
     ▼
3️⃣  Overthinking Level Analysis
     │  → Composite score from Depression + Anxiety + Panic Attack
     │  → Level classification: Low / Medium / High
     ▼
4️⃣  Machine Learning Models
     │  → Logistic Regression (baseline)
     │  → Random Forest (primary model)
     │  → Feature importance analysis
     ▼
5️⃣  Results & Insights
```

---

## 📈 EDA Highlights

### 🎂 Age Distribution
- Most students are aged **18–24**
- Peak age: **18 years old**

### ⚧ Gender Split
- Dataset includes both **Male** and **Female** students
- Female students show slightly higher reported rates of Depression and Anxiety

### 📚 Academic Year
- Responses from **Year 1 to Year 4**
- Year 1 and Year 2 students are the most represented

### 🎓 CGPA Distribution
- Most students fall in the **3.00–3.49** and **3.50–4.00** CGPA ranges

### 🧠 Mental Health Prevalence
| Condition | Approximate Prevalence |
|---|---|
| Depression | ~35% of students |
| Anxiety | ~35% of students |
| Panic Attack | ~33% of students |

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression (Baseline)
- **Target Variables:** Depression, Anxiety, Panic Attack (binary)
- **Approach:** Binary classification per mental health condition
- **Features Used:** Age, CGPA, Year of Study, Gender, Course

### 2️⃣ Random Forest (Primary Model)
- **Advantage:** Handles non-linear relationships in small datasets
- **Feature Importance:** Identifies key predictors of mental health issues
- **Multiple targets:** Separate models for each condition

### 3️⃣ Overthinking Level Prediction
- **Custom feature engineering:** Combined score from all 3 mental health indicators
- **Output:** Overthinking Level — Low / Medium / High

### Algorithm Comparison

| Algorithm | Difficulty | Best For | Recommended? |
|---|---|---|---|
| Logistic Regression | ⭐ Easy | Baseline / small data | ✅ Yes |
| Random Forest | ⭐⭐ Easy | General classification | ✅ Yes |
| Decision Tree | ⭐ Easy | Interpretability | ✅ Optional |
| SVM | ⭐⭐⭐ Medium | Small well-separated data | ✅ Optional |
| Naive Bayes | ⭐ Easy | Quick probabilistic | ✅ Optional |
| KNN | ⭐ Easy | Non-linear patterns | ⚠️ Slow prediction |
| XGBoost | ⭐⭐⭐⭐ Hard | Max accuracy | ❌ Overkill |
| Neural Network | ⭐⭐⭐⭐ Hard | Large datasets | ❌ Not suitable |

---

## 💡 Key Findings

- **Depression, Anxiety, and Panic Attacks** are highly correlated among students — students reporting one condition often report others.
- **Year of Study** has a noticeable impact — senior-year students tend to show different mental health profiles than freshers.
- **CGPA** shows a subtle inverse relationship with anxiety levels.
- **Female students** in this dataset reported slightly higher rates of mental health conditions.
- A **small dataset (101 records)** is best suited for simpler models like Logistic Regression and Random Forest over deep learning approaches.

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13 | Core programming language |
| Pandas | 2.3.3 | Data manipulation & cleaning |
| NumPy | 2.4.0 | Numerical operations |
| Scikit-Learn | Latest | Machine learning models |
| Matplotlib | Latest | Data visualization |
| Seaborn | Latest | Statistical plots |
| Jupyter Notebook | Latest | Interactive development |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/agarwalganesh/StudentOverthinkingEDA-project.git
   cd StudentOverthinkingEDA-project
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Run notebooks in order:**
   - `1️⃣ Clean_Mental_Health_Data.ipynb` — Clean the raw data
   - `2️⃣ EDA_Analysis.ipynb` — Explore the data
   - `3️⃣ Overthinking_Level_Prediction.ipynb` — Analyze overthinking levels
   - `4️⃣ Logistic_Regression.ipynb` — Baseline ML model
   - `5️⃣ Random_Forest.ipynb` — Primary ML model

---

## 👤 Author

**Ganesh Agarwal**  
📌 Data Science & Machine Learning Enthusiast  
🔗 GitHub: [@agarwalganesh](https://github.com/agarwalganesh)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
