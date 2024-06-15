from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb
import joblib
import os

FILENAME = "titanic_voting_classifier.pkl"


def preprocess_data(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    data = data.drop(["Cabin", "Embarked", "Ticket", "Fare"], axis=1)
    if train:
        data.dropna(inplace=True)
    else:
        data["Age"] = data.Age.fillna(data.Age.mean())
        # data["Fare"] = data.Fare.fillna(data.Fare.mean())

    data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    return data


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    numeric_features = ["Age", "FamilySize"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["Pclass", "Sex"]
    categorical_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.3],
    }

    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X_train, y_train)
    return best_xgb


def metrics(y_pred: pd.Series, y_prob: pd.Series, y_test: pd.Series) -> None:
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc}")

    logloss = log_loss(y_test, y_prob)
    print(f"Log Loss: {logloss}")

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_accuracy}")

    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Matthews Correlation Coefficient: {mcc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_xgb.classes_)
    disp.plot()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.show()


# Load the dataset
data = pd.read_csv("input/train.csv")

data = preprocess_data(data)

# Features and target variable
X = data[["Pclass", "Sex", "Age", "FamilySize"]]
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if os.path.isfile(FILENAME):
    print("Loading the model")
    best_xgb = joblib.load(f"output/{FILENAME}")
else:
    best_xgb = train_model(X_train, y_train)
    joblib.dump(best_xgb, f"output/{FILENAME}")

    # Predictions
    y_pred = best_xgb.predict(X_test)
    y_prob = best_xgb.predict_proba(X_test)[:, 1]
    # metrics(y_pred, y_prob, y_test)

    # Kaggle test.csv #Score: 0.75598
    # data = preprocess_data(pd.read_csv("input/test.csv"), False)
    # X_test = data[["Pclass", "Sex", "Age", "Fare", "FamilySize"]]
    # y_pred = best_xgb.predict(X_test)

    # X_test["Survived"] = y_pred
    # X_test["PassengerId"] = data["PassengerId"]
    # X_test[["PassengerId", "Survived"]].to_csv("model_1.csv", index=False)
