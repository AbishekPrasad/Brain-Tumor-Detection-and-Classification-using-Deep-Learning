from data.download_data import download_dataset
from generators.custom_data_generator import CustomDataGenerator
from utils.data_utils import get_data_paths_and_labels
from model.build_model import build_model
from model.train import train_model, evaluate_and_plot

# Load and prepare data
train_dir, test_dir = download_dataset()
train_paths, train_labels = get_data_paths_and_labels(train_dir)
test_paths, test_labels = get_data_paths_and_labels(test_dir)

train_gen = CustomDataGenerator(train_paths, train_labels, batch_size=32)
test_gen = CustomDataGenerator(test_paths, test_labels, batch_size=32)

# Build and train model
model = build_model()
model, history = train_model(model, train_gen, test_gen)

# Save and evaluate
model.save("brain_tumor_classification_model.keras")
evaluate_and_plot(model, history, test_gen)
