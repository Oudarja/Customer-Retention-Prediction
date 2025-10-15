import pandas as pd
import joblib
import time
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 

# suppress info/warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(path):
    return pd.read_csv(path)

def save_model_or_data(obj, path):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)  # save as CSV
        print(f"DataFrame saved at {path}")
    elif isinstance(obj, Model) or path.endswith('.h5'):
        obj.save(path)
        print(f"Keras model saved at {path}")
    else:
        joblib.dump(obj, path)
        print(f"Object saved at {path}")

def load_model(path):
    if path.endswith('.h5'):
        return load_model(path)
    else:
        return joblib.load(path)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f}s")
        return result
    return wrapper