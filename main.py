from src.data_preprocessing import  encode_features, scale_features_stdscale, scale_features_minmax
from src.feature_engineering import add_features
import pandas as pd
from src.model_evaluation import evaluate_model, evaluate_keras_model

from utils.utils import load_data, save_model_or_data,load_any_model

df = load_data('data/raw_data/dataset.csv')

df = encode_features(df, categorical_cols=['Gender', 'Promotion_Response', 'Email_Opt_In','Target_Churn'])

save_model_or_data(df, 'data/processed_data/processed_data.csv')

df = add_features(df)

# model loading 

rf = load_any_model('models/randomforest_best.pkl')
xgb = load_any_model('models/xgboost_best.pkl')
lgbm = load_any_model('models/lgbm_best.pkl')
ann = load_any_model('models/ann_best.h5')

test_x_rf= load_data('data/processed_data/X_test_final_rf.csv')
test_x_xgb= load_data('data/processed_data/X_test_final_xgb.csv')   
test_x_lgbm= load_data('data/processed_data/X_test_final_lgbm.csv')
test_x_ann= load_data('data/processed_data/X_test_final_ann.csv')
test_y= load_data('data/processed_data/y_test.csv')

evaluate_model(rf, test_x_rf, test_y, "Random Forest")

evaluate_model(xgb, test_x_xgb, test_y, "XGBoost")

evaluate_model(lgbm, test_x_lgbm, test_y, "LightGBM")

evaluate_keras_model(ann, test_x_ann, test_y, "ANN")


