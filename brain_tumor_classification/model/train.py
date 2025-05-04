import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

def train_model(model, train_gen, val_gen, epochs=10, learning_rate=0.0001):
    model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )

    return model, history

def evaluate_and_plot(model, history, val_gen):
    test_loss, test_acc = model.evaluate(val_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
