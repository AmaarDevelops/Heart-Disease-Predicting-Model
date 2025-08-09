import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("heart.csv")

# Basic checks
print(df.shape)
print(df.info())
print(df.describe())
print("Duplicates:", df.duplicated().sum())
print("Missing values:\n", df.isnull().sum())

# Target variable distribution
df['HeartDisease'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.show()

# Histograms for key numerical features
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)

plotting("Age", 1)
plotting("RestingBP", 2)
plotting("Cholesterol", 3)
plotting("MaxHR", 4)
plt.tight_layout()
plt.show()

# Fix Cholesterol and RestingBP 0 values
cholesterol_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, cholesterol_mean).round(2)

resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean).round(2)

# Categorical plots
sns.countplot(x='Sex', data=df)
plt.show()

sns.countplot(x='ChestPainType', hue='HeartDisease', data=df)
plt.show()

sns.countplot(x='FastingBS', hue='HeartDisease', data=df)
plt.show()

# Boxplot & Violin plot
sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
plt.show()

sns.violinplot(x='HeartDisease', y='Age', data=df)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

# Train-test split & scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred), 4)
    })

# Show results
results_df = pd.DataFrame(results)
print(results_df)

# Save best model (example: KNN)
import joblib
joblib.dump(models['KNN'], 'KNN_heart.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')
