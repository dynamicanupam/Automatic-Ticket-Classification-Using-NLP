# Automatic-Ticket-Classification-Using-NLP

## Overview
This project aims to classify customer complaints into different topics using machine learning techniques. The dataset contains text data related to customer complaints, and the goal is to predict the category or topic of each complaint. Various models such as Logistic Regression, Decision Tree, and Random Forest have been evaluated for their effectiveness in classifying the complaints accurately.

## Dataset Description
The dataset consists of customer complaints categorized into several topics:
- Bank account services
- Credit Card/Prepaid Card
- Mortgages/loans
- Theft/Dispute reporting
- Others

Each complaint is associated with text data that describes the issue faced by the customer.

## <b>Approach</b>

1.	Data Loading & Preprocessing: Conversion of .json data to a dataframe, followed by text cleaning and preprocessing.
2.	Exploratory Data Analysis (EDA): Detailed analysis to understand the data distribution and extract meaningful insights.
3.	Feature Extraction & Topic Modeling: Use of Non-Negative Matrix Factorization (NMF) to identify patterns and categorize complaints.
4.	Model Building: Training multiple supervised learning models including logistic regression, decision tree, random forest, and naive Bayes.
5.	Model Evaluation: Comparison of models based on accuracy and other evaluation metrics to select the best-performing model.

## Key Methodology
### Data Preprocessing
- Text cleaning: Tokenization, removing stopwords, punctuation, and stemming/lemmatization.
- Vectorization: Transforming text data into numerical features using TF-IDF vectorization.

### Model Building
1. **Logistic Regression**
   - Model trained using `LogisticRegression` from scikit-learn.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
2. **Decision Tree**
   - Model trained using `DecisionTreeClassifier` with and without hyperparameter tuning.
   - Hyperparameters tuned: `max_depth`, `min_samples_leaf`, `criterion`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
3. **Random Forest**
   - Model trained using `RandomForestClassifier` with and without hyperparameter tuning.
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
4. **Naive Bayes (Optional)**
   - Model trained using `MultinomialNB`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   - Hyperparameter tuning: Alpha parameter for Laplace smoothing.

### Model Evaluation
Several machine learning models were evaluated for their effectiveness in classifying complaints:

![image](https://github.com/user-attachments/assets/0e3f3f3e-d518-4692-9999-609164cf81a1)

### Model Selection
- **Logistic Regression** performed the best overall with the highest test accuracy of 88.37%, indicating robust performance in classifying customer complaints.
- **Decision Tree** showed improvement after hyperparameter tuning but did not match Logistic Regression's performance.
- **Random Forest** and **Naive Bayes** models demonstrated lower accuracies compared to Logistic Regression and Decision Tree.

## <b>Key Libraries and Frameworks</b>

•	Pandas: For data manipulation and analysis.

•	NumPy: For numerical computations.

•	Scikit-learn: For machine learning algorithms, including logistic regression, decision tree, random forest, and naive Bayes.

•	NLTK: For natural language processing tasks such as text preprocessing.

•	SpaCy: For advanced NLP tasks and text processing.

•	Matplotlib & Seaborn: For data visualization and exploratory data analysis.

•	Scikit-learn: For feature extraction, model building, and evaluation.

•	Non-Negative Matrix Factorization (NMF): For topic modeling and identifying patterns in text data.

## Conclusion
Based on the evaluation results, **Logistic Regression** is recommended for predicting customer complaint topics due to its superior performance in both training and test sets. Further optimizations could involve:
- Exploring additional text preprocessing techniques.
- Collecting more diverse complaint data to enhance model generalization.
- Experimenting with ensemble methods or deep learning architectures for potentially better performance.
