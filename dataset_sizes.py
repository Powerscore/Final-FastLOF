import os
from experiments import load_dataset

data_dir = 'data'
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    try:
        X, y = load_dataset(filepath)
        print(f"{filename}: {X.shape}")
    except Exception as e:
        print(f"{filename}: Error - {str(e)[:50]}")

