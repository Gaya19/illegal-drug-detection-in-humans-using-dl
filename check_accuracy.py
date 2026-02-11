import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xception_model.h5")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")

# ===============================
# Load model
# ===============================
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ===============================
# Load validation dataset
# ===============================
val_datagen = ImageDataGenerator(rescale=1.0 / 255)  #.0

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(299, 299),   # 🔥 must match your training size
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# Evaluate model
# ===============================
loss, acc = model.evaluate(val_gen, verbose=1)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")
