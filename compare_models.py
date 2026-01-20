# Model Comparison Script with Voting and Stacking
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Preprocessing
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)
df['BMI_Category'] = df['BMI'].apply(lambda x: 0 if x<18.5 else (1 if x<25 else (2 if x<30 else 3)))
df['Age_Group'] = df['Age'].apply(lambda x: 0 if x<30 else (1 if x<45 else (2 if x<60 else 3)))

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data first for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()

# Voting Classifier (soft voting - uses probabilities)
voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    voting='soft'
)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# All models to compare
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('Voting Classifier', voting_clf),
    ('Stacking Classifier', stacking_clf)
]

print()
print("=" * 60)
print("    MODEL COMPARISON (5-Fold Cross-Validation on Scaled Data)")
print("=" * 60)

results = []
for name, model in models:
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results.append((name, cv_scores.mean(), cv_scores.std()))
    print(f"  {name:25} | {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

print("=" * 60)
# Sort by accuracy
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
print("  RANKING:")
for i, (name, acc, std) in enumerate(results_sorted, 1):
    marker = ">>>" if i == 1 else "   "
    print(f"  {marker} {i}. {name}: {acc:.4f}")
print("=" * 60)
print(f"  BEST MODEL: {results_sorted[0][0]}")
print(f"  Accuracy: {results_sorted[0][1]:.4f} +/- {results_sorted[0][2]:.4f}")
print("=" * 60)
print()
