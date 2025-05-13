Table of Contents
Data Cleaning and Transformation........................................................................................ 4
Checking Dataset Shape .................................................................................................... 4
Removing Irrelevant Columns........................................................................................... 4
Handling Missing Values .................................................................................................. 5
Verifying Data Types........................................................................................................ 5
Analyzing Numerical Features .......................................................................................... 5
Checking Final Dataset Shape ........................................................................................... 5
Data Visualization................................................................................................................. 6
Model Building ..................................................................................................................... 8
Selected Predictors............................................................................................................ 7
Linear Regression model................................................................................................... 9
Part 2: Improved model....................................................................................................... 11
Data Cleaning and Transformation
Dataset preparation is as important as building a predictive model, as it ensures the quality and
reliability of the dataset before building a predictive model. The “patients.csv” provides various
features about patient health conditions and treatment. This section goes through all the steps
made to clean and transform data in order to remove inconsistencies and handle missing values
in order to improve model performance.
Checking Dataset Shape
The initial shape of the dataset is:
 Number of Rows: 200,031
 Number of Columns: 22
Understanding the dataset's structure is essential to assess its complexity and plan the cleaning
process.
Removing Irrelevant Columns
Some columns provide little to no value for predictive modeling. The following columns were
removed with justification:
 MEDICAL_UNIT: Represents the medical unit handling the patient, which does not
directly affect patient readmission.
 USMER: Indicates whether the medical unit belongs to the Mexican health system;
this is not relevant to the clinical factors under investigation.
 DATE_DIED: This column is only relevant for mortality analysis, not predicting
readmission.
After dropping these columns, the dataset has 19 columns.
Handling Missing Values
Several columns contained missing values represented by the symbol "?". These were
converted to NaN for easier handling. Additionally:
 Missing Value Summary (Before):
o Columns like "PREGNANT" and "ICU" had noticeable missing values.
o Binary columns (e.g., "INTUBED") with values other than 1 or 2 were treated
as missing data.
 Approach: Rows with missing values were dropped to ensure data integrity. An
alternative approach could be imputing values, but for this model, a more aggressive
strategy was adopted to avoid introducing bias.
 Missing Value Summary (After):
o All rows with missing data were removed, significantly reducing the dataset
size but improving its reliability.
Verifying Data Types
The dataset was inspected to ensure appropriate data types for each column:
 Numerical columns like "AGE" remained as integers.
 Binary features (e.g., "DIABETES," "PNEUMONIA") were converted to integers for
model compatibility.
Analyzing Numerical Features
Summary statistics were computed to identify potential outliers. Notable findings:
 AGE: Some extreme values were present, indicating potential data entry errors.
Outliers were removed based on reasonable age ranges.
 Normalization: Features like "AGE" were normalized using Min-Max scaling to
ensure fair model input.
Checking Final Dataset Shape
After preprocessing, the dataset size changed to:
 Final Number of Rows: Reduced after handling missing values and outliers.
 Final Number of Columns: 19
Data Visualization
Data cleaning is an essential step in data preparation in ML, and after cleaning the data, we
can make a visual representation to identify patterns and relationships of the features. This
section focuses on primary figures in order to understand the potential of the given dataset in
creating predictions (Cao et al., 2021).
1. Distribution of Target Variable
Moreover, the dependent variable, “CLASIFICATION_FINAL”, is the dependent variable that
shows whether COVID-19 is confirmed or not at last. The following plots its distribution so
that it is possible to identify whether the data is imbalanced since this would have an impact
on the model.
 Insight: A balanced dataset increases the model’s training, but otherwise, may need to
use, for instance, sampling techniques.
2. ICU Cases vs Age
Visualizing the number of ICU admissions across different age groups helps determine if age
influences ICU admission rates.
 Insight: Older age groups may have higher ICU admissions, indicating age as a
potential risk factor.
3. Target Variable vs "CLASIFICATION_FINAL"
Plotting the target variable count against "CLASIFFICATION_FINAL" reveals how different
classification categories align with the target outcomes.
 Insight: Understanding the relationship between classification and outcomes helps
refine prediction strategies.
4. Scatter Matrix and Correlation Matrix
 Scatter Matrix: This plot visualizes pairwise relationships across numerical features,
revealing clusters and trends.
 Correlation Matrix: Measures linear relationships between features. Highly correlated
features could introduce multicollinearity, impacting model stability.
 Insight: Identifying highly correlated features allows the consideration of
dimensionality reduction techniques such as PCA (Principal Component Analysis).
5. Additional Plots
To gain further insights, additional visualizations were generated:
 Age Distribution: This shows the overall spread of patient ages.
 Comorbidity Analysis: Counts patients with multiple comorbidities like diabetes,
hypertension, and obesity.
 ICU Admissions by Comorbidities: Highlights the effect of comorbid conditions on
ICU admissions.
Model Building
Selected predictors
The selected predictors are centered on the risk factors that may lead to the admittance of a
patient to the ICU. P vital since patients of a certain age are more vulnerable to complications.
SEX and PREGNANT define gender-based risks encountered by specific categories of
patients; on the other hand, PATIENT_TYPE identifies the severity of afflictions. Pre-existing
conditions like DIABETES, COPD, ASTHMA, HIPERTENSION, OBESITY, and
CARDIOVASCULAR increase complication risks. PNEUMONIA and INTUBED are two
meshes that underscore what would be causing respiratory distress and frequently requiring
intensive care. CLASIFFICATION_FINAL contains clinical severity that offers extra
information and predicts the patient's state. In combination, these features give the model a
wide vision about the shape of every patient’s health and help in distinguishing high-risk cases
and timely intervention from physicians.
Linear Regression model
To establish the predictors that can be used in a logistic regression model to determine
patients at risk of being admitted to the ICU, the following criteria were used. The next step
was to divide the entire data sample into the training data and the test data, and the ratio used
was 80:20. To avoid overfitting and come up with a fair estimation of the efficiency of the
learning algorithm, cross-validation was used (Ambrish et al., 2022).
Two performance indices were used to evaluate the model: accuracy, which gave an overall
percentage of correct labels, while the ROC AUC displayed the model’s capability of
separating ICU and non-ICU patients. Furthermore, the utilization of the Confusion Matrix and
Classification Report offered an explanation towards the level of precision of the model, its
recall, as well as its F1-score. The training of the initial model allowed for the evaluation of
the effectiveness of the program, as well as demonstrating the effects of data skewness and
pointing out possible improvements. This approach was effective for a thorough evaluation of
the model, which in turn provided the steps to follow in the subsequent modeling process to
increase the correct prediction of the ICU admissions under the minority class.
Especially when the proportion is as small as 7.5%, which means that the data is imbalanced.
Again, having a better insight about which performance metrics to consider, other than
accuracy. These are the following metrics that the model is checked for:
 Accuracy: Measures the overall correctness of classifications, but it is rather deceptive
in case of unbalanced sources. In this case, a high level of accuracy of most patients as
non-ICU may obscure the identification of high-risk patients.
 Achievement: Concerns to do with the number of patients that were forecasted to be
admitted into the ICU and the number that was actually admitted. High precision
reduces the number of false positives, and this is valuable when there are few ICUs
where patients can be admitted.
 Recall (Sensitivity): Measures the model’s ability to correctly identify ICU cases. High
recall ensures that fewer critical cases are missed, making it essential in healthcare
contexts.
 F1-Score: Provides a balance between precision and recall, offering a more
comprehensive measure of performance, especially when classes are imbalanced.
 ROC AUC (Receiver Operating Characteristic - Area Under Curve): Measures the
quality of the model to classify patients at fulfilling all the criteria of an ICU and all the
criteria of a non-ICU.
 Confusion Matrix: This provides more detailed quantitative information that shows
the number of correct predictions made on the actual positive, incorrect predictions
made on the actual positive, correct predictions made on the actual negative and lastly,
predictions made on the actual negatives that are incorrect.
In order to address the issue of the frame class imbalance the following method was applied so
as to balance the frame datasets through synthetic minority class samples; SMOTE The process
of resampling consisted out of random selecting cases from the ICU and non-ICU groups and
conducting the training in order to have a better training set containing equally intensively
trained ICU and non-ICU cases. Thus, the above-mentioned resampled data were used to train
a new logistic regression model and to test it on the original test data. These positive changes
were noted in Recall and F1_Score on the balanced model, and, therefore, it could identify
more high-risk ICU patients. When it comes to distinguishing between patients who were and
were not in ICU, an increase in the ROC AUC shows the improvement. This contributed to the
improvement of the model’s ability to identify a number of instances in the minority class. The
formalization of additional evaluation measures has shown the need to address the imbalanced
data issue and the need to minimize the risk of treating critically ill patients to increase the
prediction’s accuracy.

Part 2: Improved model
1. Improved Classification Model with PySpark
 Load the full dataset into a PySpark DataFrame.
 Apply data cleaning and transformation, handling missing values through
imputation rather than dropping them (DeZyre, 2021).
 Perform feature engineering to enhance predictive power.
 Train a classification model, such as Logistic Regression, Random Forest, or
Gradient-Boosted Trees, using Spark MLlib.
 Evaluate the model using ROC AUC, Precision, Recall, and F1-score and compare it
to the previous model.
2. Clustering with K-Means
 Apply K-Means clustering to group similar patients (Khan et al., 2021).
 Compare the clusters with the original ICU classification and visualize the results.
 Analyze cluster characteristics to justify the segmentation.
3. Local Classifiers for Clusters
 Train individual classifiers for each cluster.
 Compare the cluster-based classifiers with the main classification model.
 Discuss whether localized models improve prediction accuracy.
4. Balancing Data and Retraining the Model
 Use SMOTE or other oversampling/undersampling techniques to balance the
dataset.
 Retrain and evaluate the model using the balanced dataset.
 Compare performance improvements over the unbalanced model
