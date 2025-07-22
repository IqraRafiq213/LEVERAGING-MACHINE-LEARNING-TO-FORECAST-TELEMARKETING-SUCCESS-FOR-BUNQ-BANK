**Objective **
Improve telemarketing effectiveness in acquiring long-term deposit clients for Bunq, a digital bank operating in a competitive fintech landscape. By a machine learning-powered decision
support system to identify high-potential customer, that satisfies the requirements of maximising recall, reducing targeting inefficiencies, and ensuring model transparency in order to help Bunq’s marketing team
optimise outreach customer strategies and efficiently achieve long-term deposit growth objectives. 
**Model Evaluation**
The project focuses on building a machine learning pipeline to predict long-term deposit clients by using the UCI Bank Marketing dataset1. Data Exploration: The first phase of the
project consists of loading and initial exploratory analysis, where feature distribution and summary statistics were reviewed. It was observed that the distribution of features like balance, duration and campaign was highly
skewed. To address this, logarithmic transformations were applied (log1p) for strictly positive features containing zeros and a signed logarithmic transformation for features containing negative values. There are eight
categorical variables present in the datasets and separated into ordinal, nominal and numeric types for appropriate preprocessing. The days of the week are renamed after days of the month as the feature contains values ranging from 1 to
30. The dataset was split into training and test sets, with 20% reserved as the test set. In order to ensure the class distribution, stratification is applied to the target variable.
**Data Preprocessing:** At this stage, features were implemented using scikit Learn pipeline and column transformer. Ordinal features such as ‘education’ were encoded using an ordinal
Encoder, nominal features were handled with one-hot encoding, and numerical features were standardised after imputation. The imputation strategy for the numerical and ordinal features includes ‘mean’ and ‘unknown’, respectively.
The target variable contains binary values, which are later converted into ‘yes’ as 1 and ‘no’ as 0.
**Model Evaluation:** Multiple classifiers were trained and evaluated: Logistic Regression (with class balancing), Random Forest, Support Vector Classifier (SVC), Neural Network and a
Stochastic Gradient Descent (SGD) classifier using log-loss. Given that the business goal is to prioritise the number of positive responses, recall was prioritised over precision. Models were
assessed using 3-fold cross-validation, with performance metrics including confusion matrices, precision, recall, accuracy, ROC AUC, and precision-recall curves. For this case, PR (Precision-Recall) curve and ROC AUC are
considered as evaluation metric due to the imbalanced nature of the data. By using ROC AUC, Random Forest showed the most balanced performance. However, the Logistic regression model demonstrated the highest
recall among all models.
