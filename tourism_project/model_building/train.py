import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import mlflow
import os

# Set MLflow tracking URI (can be a local directory or a remote server)
# For GitHub Actions, we'll use a local directory that gets committed
# to allow the UI to inspect runs. For local, we can use a server.
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment for local MLflow server

# Load the datasets
X_train = pd.read_csv('Xtrain.csv')
X_test = pd.read_csv('Xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze() # .squeeze() to convert DataFrame to Series
y_test = pd.read_csv('ytest.csv').squeeze()

# Define preprocessing steps
# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['number']).columns

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
)

# Define the model pipeline
pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__max_depth': [3, 5]
}

# Set up MLflow for tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'test_size': 0.2,
        'random_state': 42,
        'stratify': 'y',
        'preprocessor': 'StandardScaler, OneHotEncoder',
        'model_type': 'XGBoost'
    })

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params({
        'best_n_estimators': grid_search.best_params_['xgbclassifier__n_estimators'],
        'best_learning_rate': grid_search.best_params_['xgbclassifier__learning_rate'],
        'best_max_depth': grid_search.best_params_['xgbclassifier__max_depth']
    })

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metrics({
        'test_accuracy': report['accuracy'],
        'test_precision': report['1']['precision'],
        'test_recall': report['1']['recall'],
        'test_f1_score': report['1']['f1-score'],
        'test_roc_auc': roc_auc
    })

    print("Best parameters:", grid_search.best_params_)
    print("Test ROC AUC:", roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    model_path = 'tourism_project/deployment/best_model.joblib'
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, "model")

    print(f"Best model saved to {model_path}")
