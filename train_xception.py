import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# Paths (Train uses augmented dataset; Val/Test use original)
# ==============================
base_path = r"C:\Users\tadip\Downloads\DrugDetectionProject_Xception\DrugDetectionProject"

train_dir = os.path.join(base_path, "augmented_dataset", "train")  # only train augmented
val_dir   = os.path.join(base_path, "dataset", "val")               # original val
test_dir  = os.path.join(base_path, "dataset", "test")              # original test
model_dir = os.path.join(base_path, "models")

os.makedirs(model_dir, exist_ok=True)

# ==============================
# Image Data Generators
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Validation and test: only rescale
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode="categorical"
)
print(train_generator.class_indices)


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# ✅ Print class labels for verification
print("\n📂 Classes detected:", train_generator.class_indices)

# Save class indices to JSON (for Flask app later)
class_indices_path = os.path.join(model_dir, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f)
print(f"✅ Class indices saved at {class_indices_path}")

# ==============================
# Build Model (Xception)
# ==============================
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3)) #224

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # reduce overfitting
predictions = Dense(len(train_generator.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# ==============================
# Compile
# ==============================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==============================
# Callbacks
# ==============================
checkpoint_path = os.path.join(model_dir, "best_xception_model.h5")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]

# ==============================
# Train (initially with frozen base layers)
# ==============================
history = model.fit(
    train_generator,
    epochs=6,
    validation_data=val_generator,
    callbacks=callbacks,
   
)

# ==============================
# Fine-tuning (unfreeze top layers)
# ==============================
for layer in base_model.layers[-50:]:  # unfreeze last 30 layers  #30
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(
    train_generator,
    epochs=6,
    validation_data=val_generator,
    callbacks=callbacks,
    
)

# ==============================
# Save Final Model
# ==============================
final_model_path = os.path.join(model_dir, "xception_model.h5")
model.save(final_model_path)
print(f"✅ Model training complete and saved as {final_model_path}")

# ==============================
# Evaluate on Test Set
# ==============================
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print("\n📊 Final Model Performance:")
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}\n")

# Save accuracy to text file
with open(os.path.join(model_dir, "final_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")

# ==============================
# Plot Training Curves
# ==============================
def plot_training(history, history_fine, out_path):
    acc = history.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history.history["loss"] + history_fine.history["loss"]
    val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b-", label="Training acc")
    plt.plot(epochs, val_acc, "r-", label="Validation acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b-", label="Training loss")
    plt.plot(epochs, val_loss, "r-", label="Validation loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

plot_training(history, history_fine, os.path.join(model_dir, "training_curves.png"))
print("📊 Training curves saved in models/training_curves.png")


