# 📊 Online User Behavior & Productivity Analysis  
### Using Machine Learning & Bayesian Graphical Models  
**Course**: Data Mining & Data Warehousing  
**Team**: Synergy  

---

##  Project Overview
This project analyzes how **online behavior and digital habits** influence user productivity.  
We use a real-world dataset containing demographic attributes, social media usage, notification exposure, sleep patterns, stress levels, and productivity scores.

The study combines:

-  Classical Machine Learning (Logistic Regression, KNN, SVM, Random Forest)  
-  Unsupervised Clustering (K-Means + PCA Visualization)  
-  Bayesian Network Structure Learning (pgmpy) for probabilistic dependency discovery  
-  ROC, Confusion Matrix, Feature Importance, and Model Comparison

This makes the project **not only predictive**, but also **explainable** — which is critical in behavioral analytics.

---

##  Dataset Description
Each row represents one user (cross-sectional dataset).  
Key features include:

| Category | Example Features |
|----------|------------------|
| Demographics | age, gender, job_type |
| Online activity | daily_social_media_time, number_of_notifications, social_platform_preference |
| Productivity | perceived_productivity_score, actual_productivity_score |
| Digital wellbeing | uses_focus_apps, has_digital_wellbeing_enabled |
| Health & lifestyle | sleep_hours, stress_level, coffee_consumption_per_day |

A binary target variable **`HighProductivity`** is created from `actual_productivity_score`.

---

##  Tech Stack
| Component | Technologies |
|-----------|--------------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Graphical Model | pgmpy (Bayesian Network) |
| Clustering | K-Means, PCA |
| Saving Model | joblib |

---

##  Methodology Pipeline
1. **Data Cleaning & Imputation**  
   - Missing numerical → Median  
   - Missing categorical → Most Frequent  
   - Duplicates removed & column standardization

2. **Encoding & Scaling**  
   - LabelEncoder for category → numeric conversion  
   - StandardScaler for uniform feature scaling

3. **Exploratory Data Analysis**
   - Distributions, correlations, histograms
   - Target imbalance check

4. **Unsupervised Clustering**
   - Elbow method to estimate optimal `k`
   - K-Means clustering
   - PCA for 2D visualization of clusters

5. **Model Training & Hyperparameter Tuning**
   - Logistic Regression  
   - KNN  
   - SVM  
   - Random Forest  
   - GridSearchCV + StratifiedKFold

6. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC curves  
   - Confusion matrices  
   - Feature importance chart

7. **Bayesian Network (Graphical Model)**
   - Numeric discretization
   - Hill-Climb Structure Learning with BIC / K2 scoring
   - Network visualization using NetworkX

8. **Model Saving**
   - Best model exported as `.joblib` file

---

##  Results & Findings
- **Random Forest** achieved the strongest predictive performance in terms of F1 and ROC-AUC.
- **Top influential features** included: 
  - social media usage duration  
  - notifications count  
  - sleep hours  
  - stress level  
  - perceived productivity

- The **Bayesian Network** revealed meaningful dependencies such as:  
  `perceived_productivity_score → actual_productivity_score → HighProductivity`

This shows that **subjective perception strongly correlates with measurable productivity**, and both are influenced by lifestyle factors.

---

##  Bayesian Network Visualization Example
When pgmpy is available, the notebook generates:

This directed graph shows conditional dependencies between key variables.

---

##  Limitations
- Dataset is **cross-sectional**, not time-series → HMM skipped
- Causality **cannot** be proven, only probabilistic influence
- Dependent on self-reported productivity and behavior

---

##  Future Work
- Collect session/timestamp-level data for sequential modeling (HMM/LSTMs)
- Add SHAP explainability for model transparency
- Deploy as a web app using Streamlit or Flask
- Expand dataset size and demographic diversity

---

##  Running the Project
###  Requirements
- ```pip install -r requirements.txt```

### Run the Notebook
1. Place dataset CSV in the project folder  
2. Open notebook in Jupyter/Colab  
3. Run all cells sequentially  
4. Outputs such as ROC plots, confusion matrices, feature importance, and BN graph will be saved automatically.

---

##  Acknowledgements
- NSL-KDD, UCI ML Repository, pgmpy, sklearn docs  
- Course: **Data Mining & Data Warehousing**

---

### ⭐ If you find this project useful, consider giving the repository a star!
