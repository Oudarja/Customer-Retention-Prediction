from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
import os
import matplotlib.pyplot as plt
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print('------------------------------------')
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print('------------------------------------')
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('------------------------------------')
    # --- Plot and save ROC curve ---
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.title(f"ROC Curve - {model_name}")
    
    save_path = f"fig/{model_name}_roc_curve.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")

def evaluate_keras_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype("int32")
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print('------------------------------------')
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print('------------------------------------')
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('------------------------------------')
    # --- Plot and save ROC curve ---
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.title(f"ROC Curve - {model_name}")
    
    save_path = f"fig/{model_name}_roc_curve.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")