import os
import numpy as np

def get_data_paths_and_labels(directory):
    class_names = sorted(os.listdir(directory))
    file_paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        for fname in os.listdir(class_path):
            file_paths.append(os.path.join(class_path, fname))
            one_hot = np.zeros(len(class_names))
            one_hot[idx] = 1
            labels.append(one_hot)
    return file_paths, labels
