import kaggle
import pandas as pd
import os

RAW_DIR = 'data/raw'
SUBSAMPLE_SIZE = 500_000

# def download_dataset():
#     os.makedirs(RAW_DIR,exist_ok=True)
#     kaggle.api.dataset_download_files(
#         "thoughtvector/customer-support-on-twitter",
#         path=RAW_DIR,
#         unzip=True
#     )

def create_subsample():
    full_path = os.path.join(RAW_DIR,'twcs/twcs.csv')
    df = pd.read_csv(full_path,nrows=SUBSAMPLE_SIZE)
    df.to_csv(os.path.join(RAW_DIR,"twcs_subsample.csv"),index=False)
    print(f"Subsample saved: {len(df)} rows")

if __name__ == "__main__":
    # download_dataset()
    create_subsample()