import os
import matplotlib.pyplot as plt
import cv2   # ✅ Added for eye detection


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def plot_training_curves(history, out_path="training_curves.png"):
    hist = history.history
    # Accuracy plot
    plt.figure()
    plt.plot(hist["accuracy"], label="train_acc")
    plt.plot(hist["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_acc.png"))
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_loss.png"))
    plt.close()


# ================================
# ✅ EYE DETECTION FUNCTION ADDED
# ================================
def contains_eye(image_path):
    """Check if uploaded image contains at least one human eye."""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Return True if at least one eye found
    return len(eyes) > 0
