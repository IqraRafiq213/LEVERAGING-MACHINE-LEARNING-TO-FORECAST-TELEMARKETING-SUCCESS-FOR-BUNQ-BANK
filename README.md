# 📊 Leveraging Machine Learning to Forecast Telemarketing Success for Bunq Bank

> **Can we predict which customers will open a long-term deposit — before we even call them?**  
> This project builds a machine learning pipeline to do exactly that, helping Bunq's marketing team target smarter and spend less.

---

##  Business Context

Bunq is a digital-first fintech bank operating in a competitive European landscape. Like many banks, it runs telemarketing campaigns to acquire long-term deposit clients — customers who provide stable, predictable capital over time.

The problem: campaigns consume significant internal resources while conversion rates remain low. Cold-calling at scale is expensive and inefficient.

**The goal of this project** is to build a decision-support system that identifies high-potential customers *before* they're contacted, by learning patterns from historical campaign data. The system prioritises **recall** — catching as many true positives as possible — to reduce missed opportunities, while maintaining transparency for marketing teams.

---

## Repository Structure

```
├── portfolio_copy.ipynb   # Full end-to-end ML pipeline (EDA → preprocessing → modelling → evaluation)
└── README.md
```

---

## 📦 Dataset

The project uses the **UCI Bank Marketing Dataset**, which contains data from direct marketing campaigns (phone calls) of a Portuguese banking institution.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Size:** ~45,000 records, 17 features
- **Target variable:** `y` — whether the client subscribed to a long-term deposit (`yes` / `no`)
- **Class imbalance:** The dataset is imbalanced, with a minority of positive responses (~11%)

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Reviewed feature distributions and summary statistics
- Identified heavy right-skew in `balance`, `duration`, and `campaign`
- Applied **log transformations** (`log1p` for non-negative features; signed log for features containing negative values) to reduce skewness
- Categorised 8 categorical variables into ordinal, nominal, and numeric types for appropriate preprocessing
- Corrected the `day` feature (renamed from "day of week" to "day of month", values 1–30)

### 2. Data Preprocessing
Built a **scikit-learn Pipeline** with `ColumnTransformer` to handle each feature type:

| Feature Type | Strategy |
|---|---|
| Numerical | Impute with `mean` → StandardScaler |
| Ordinal (e.g. `education`) | Impute with `'unknown'` → OrdinalEncoder |
| Nominal | Impute with `'unknown'` → OneHotEncoder |

The target variable was binarised: `yes → 1`, `no → 0`.  
An 80/20 stratified train-test split was used to preserve class balance.

### 3. Model Training & Evaluation

Five classifiers were trained and evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Class-weight balanced; highest recall |
| Random Forest | Best overall ROC AUC |
| Support Vector Classifier (SVC) | Strong boundary-based classifier |
| Neural Network (MLP) | Non-linear patterns |
| SGD Classifier | Log-loss; scalable baseline |

**Evaluation metrics used:**
- Confusion matrix
- Precision, Recall, Accuracy
- **ROC AUC** — primary metric for imbalanced data
- **Precision-Recall (PR) curve** — better suited when positive class is rare

> 🏆 **Random Forest** achieved the most balanced performance by ROC AUC.  
> 🎯 **Logistic Regression** achieved the highest recall — prioritised here given the business goal of minimising missed deposit opportunities.

Cross-validation used 3-fold stratified CV to ensure reliable estimates.

---

##  Tech Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

##  Getting Started

```bash
# Clone the repository
git clone https://github.com/IqraRafiq213/LEVERAGING-MACHINE-LEARNING-TO-FORECAST-TELEMARKETING-SUCCESS-FOR-BUNQ-BANK.git
cd LEVERAGING-MACHINE-LEARNING-TO-FORECAST-TELEMARKETING-SUCCESS-FOR-BUNQ-BANK

# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook portfolio_copy.ipynb
```

---

##  Key Takeaways

- **Feature engineering matters:** Log-transforming skewed features meaningfully improved model stability.
- **Recall vs. precision trade-off:** In a telemarketing context, missing a potential depositor (false negative) is more costly than calling a non-converter (false positive) — so recall was the primary optimisation target.
- **Imbalanced data requires care:** Standard accuracy is misleading here; ROC AUC and PR curves gave a more honest picture of performance.
- **Logistic Regression remains competitive:** Despite its simplicity, it outperformed more complex models on recall, making it the most business-appropriate choice.

---



---

*Data source: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, Elsevier.*
