from src.data_preprocessing import  encode_features, scale_features_stdscale, scale_features_minmax
from src.feature_engineering import add_features
import pandas as pd

from utils.utils import load_data, save_model_or_data, load_model, timer

df = load_data('data/raw_data/dataset.csv')

df = encode_features(df, categorical_cols=['Gender', 'Promotion_Response', 'Email_Opt_In','Target_Churn'])

save_model_or_data(df, 'data/processed_data/processed_data.csv')

df = add_features(df)

save_model_or_data(df, 'data/processed_data/processed_data_with_engineered_features1.csv')