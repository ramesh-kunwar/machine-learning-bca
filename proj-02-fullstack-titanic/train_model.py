import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
df = pd.read_csv('Titanic.csv')

# Preprocess the data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Fill missing values
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    # Convert categorical variables
    le = LabelEncoder()
    df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
    df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])
    
    # Create family size feature
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    # Select features for training
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
    X = df_processed[features]
    y = df_processed['Survived']
    
    return X, y

# Preprocess the data
X, y = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Perform cross-validation to check for overfitting
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train the model on the full training set
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {test_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and preprocessing information
model_data = {
    'model': model,
    'features': X.columns.tolist(),
    'feature_means': X.mean().to_dict(),
    'feature_stds': X.std().to_dict()
}

with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved as 'titanic_model.pkl'") 