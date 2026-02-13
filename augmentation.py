import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# ==============================
# Paths
# ==============================
input_dir = r"C:\drug-detection"
output_dir = r"C:\drug-detection"

# How many augmented images per class
AUGMENT_SIZE = 1000  

# ==============================
# Create output folders
# ==============================
os.makedirs(output_dir, exist_ok=True)

for cls in os.listdir(input_dir):
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# ==============================
# Define augmentation
# ==============================
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

# ==============================
# Augment class by class
# ==============================
for cls in os.listdir(input_dir):
    cls_input_path = os.path.join(input_dir, cls)
    cls_output_path = os.path.join(output_dir, cls)

    images = os.listdir(cls_input_path)
    print(f"🔄 Augmenting class: {cls} (found {len(images)} images)")

    img_count = 0
    for img_name in images:
        img_path = os.path.join(cls_input_path, img_name)
        try:
            img = load_img(img_path, target_size=(224, 224))  # resize like model input
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Generate images
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=cls_output_path,
                                      save_prefix=cls,
                                      save_format='jpg'):
                i += 1
                img_count += 1
                if i >= AUGMENT_SIZE // len(images):
                    break
        except Exception as e:
            print(f"⚠️ Skipping {img_name}: {e}")

    print(f"✅ Saved ~{img_count} augmented images for class: {cls}")

print("\n🎉 Augmentation complete! Check the folder:")
print(output_dir)

