import pandas as pd
import joblib
import time
import os
from tensorflow.keras.models import Model, load_model as keras_load_model

# suppress TensorFlow info/warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data(path):
    """Load CSV data."""
    return pd.read_csv(path)

def save_model_or_data(obj, path):
    """Save a DataFrame, Keras model, or other Python object."""
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
        print(f"✅ DataFrame saved at {path}")
    elif isinstance(obj, Model) or path.endswith('.h5'):
        obj.save(path)
        print(f"✅ Keras model saved at {path}")
    else:
        joblib.dump(obj, path)
        print(f"✅ Object saved at {path}")

def load_any_model(path):
    """Load a Keras (.h5) or joblib model."""
    if path.endswith('.h5'):
        return keras_load_model(path)
    else:
        return joblib.load(path)

def timer(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} executed in {end - start:.2f}s")
        return result
    return wrapper
