from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    return cm