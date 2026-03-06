import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Load dataset
df = pd.read_csv(r"C:\Users\VICTUS\Bank-Marketing-ML-App\data\bank.csv")
print("Original dataset shape:", df.shape)

# Convert target variable
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

# Convert categorical columns into numerical
df = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df.shape)

# Separate features and target
X = df.drop("deposit", axis=1)
y = df["deposit"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("After SMOTE shape:", X_train_bal.shape)

# Calculate imbalance ratio for XGBoost
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=4000),
    "Decision Tree": DecisionTreeClassifier(max_depth=12),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
}

results = []
scaler = StandardScaler()

# Train and evaluate each model
for name, model in models.items():

    print("\n-----------------------------")
    print("Model:", name)

    # Logistic Regression needs scaling
    if name == "Logistic Regression":
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train_bal)
        predictions = model.predict(X_test_scaled)

    # XGBoost uses original training data
    elif name == "XGBoost":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # Tree models use SMOTE data
    else:
        model.fit(X_train_bal, y_train_bal)
        predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("Confusion Matrix:\n", cm)

    results.append([name, acc, prec, rec])

    joblib.dump(model, name + ".pkl")

    


# Create comparison table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall"])

print("\nFinal Comparison:")
print(results_df)

# Find best models
best_acc = results_df.loc[results_df["Accuracy"].idxmax()]
best_prec = results_df.loc[results_df["Precision"].idxmax()]
best_rec = results_df.loc[results_df["Recall"].idxmax()]

print("\nBest Accuracy:", best_acc["Model"])
print("Best Precision:", best_prec["Model"])
print("Best Recall:", best_rec["Model"])

results_df.to_csv("model_results.csv", index=False)