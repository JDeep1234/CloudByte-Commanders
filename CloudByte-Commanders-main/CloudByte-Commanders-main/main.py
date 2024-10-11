import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Flatten
from tensorflow.keras.models import Model

# Load student and teacher datasets
students_data = pd.read_csv('student_prediction.csv') #Please keep the csv files in the same folder as main.py
teachers_data = pd.read_csv('Teacher_survey.csv')

# Assuming 'StudentID' is a common identifier in both datasets
common_column = 'Course ID'

# Merge datasets without specifying a common column
merged_data = pd.merge(students_data, teachers_data, how='inner', on=common_column)

# List of columns containing teacher responses
teacher_response_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9','q10']

# Concatenate teacher responses into a single column
merged_data['TeacherResponses'] = merged_data[teacher_response_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Feature extraction using TfidfVectorizer for teacher responses
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(merged_data['TeacherResponses'])

# Feature engineering for student attributes
numeric_features = ['STUDY_HRS', 'READ_FREQ', 'READ_FREQ_SCI', 'ATTEND_DEPT',
                    'PREP_STUDY', 'PREP_EXAM',  'CUML_GPA', 'EXP_GPA','GRADE']



# Preprocessing for numerical and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
       
    ])

# Combine teacher and student features
X_teacher = X_text.toarray()
X_student = preprocessor.fit_transform(merged_data)

# Neural Network Model
input_teacher = Input(shape=(X_teacher.shape[1],))
input_student = Input(shape=(X_student.shape[1],))

# Embedding layer for teacher responses
embedding_teacher = Embedding(input_dim=X_teacher.shape[1], output_dim=32, input_length=X_teacher.shape[1])(input_teacher)
flatten_teacher = Flatten()(embedding_teacher)

# Concatenate teacher and student features
concatenated_features = Concatenate()([flatten_teacher, input_student])

# Dense layers for prediction
dense1 = Dense(128, activation='relu')(concatenated_features)
dense2 = Dense(64, activation='relu')(dense1)
output_layer = Dense(1, activation='linear')(dense2)

# Model definition
model = Model(inputs=[input_teacher, input_student], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Split the data into training and testing sets
X_train_teacher, X_test_teacher, X_train_student, X_test_student, y_train, y_test = train_test_split(
    X_teacher, X_student, merged_data['GRADE'], test_size=0.2, random_state=42)

# Train the model
model.fit([X_train_teacher, X_train_student], y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
mae = model.evaluate([X_test_teacher, X_test_student], y_test)
print(f'Mean Absolute Error: {mae}')

# Make predictions on test data
predictions = model.predict([X_test_teacher, X_test_student])

# Calculate MAE for the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Assuming you have predictions and y_test
print(f'Length of STUDENTID: {len(merged_data["STUDENTID"])}')
print(f'Length of TEACHERID: {len(merged_data["TEACHERID"])}')
print(f'Length of Predictions: {len(predictions.flatten())}')
print(f'Length of True Grades: {len(y_test)}')
print(predictions.shape)
print(merged_data.index)
print(predictions.index)
print(y_test.index)


# Reset indices to ensure alignment
merged_data = merged_data.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

## Scatter plot of True Grade vs. Predicted Grade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='True_Grade', y='Predicted_Grade', data=mapping_table, hue='Teacher_ID')
plt.title('True Grade vs. Predicted Grade')
plt.xlabel('True Grade')
plt.ylabel('Predicted Grade')
plt.legend(title='Teacher ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Bar plot of Grade Improvement Percentage
plt.figure(figsize=(10, 6))
sns.barplot(x='Teacher_ID', y='Grade_Improvement_Percentage', data=mapping_table)
plt.title('Grade Improvement Percentage by Teacher')
plt.xlabel('Teacher ID')
plt.ylabel('Grade Improvement Percentage')
plt.show()

# Box plot of Grade Improvement Percentage by Teacher
plt.figure(figsize=(10, 6))
sns.boxplot(x='Teacher_ID', y='Grade_Improvement_Percentage', data=mapping_table)
plt.title('Grade Improvement Percentage by Teacher')
plt.xlabel('Teacher ID')
plt.ylabel('Grade Improvement Percentage')
plt.show()
