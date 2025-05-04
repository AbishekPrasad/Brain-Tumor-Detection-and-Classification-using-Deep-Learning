import os
import kagglehub

def download_dataset():
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    return os.path.join(path, "Training"), os.path.join(path, "Testing")
