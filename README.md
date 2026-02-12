# 👁️ Drug Detection using Ocular Images (Xception CNN)

This project detects whether an eye image is:
- Drug affected
- Infected
- Normal

Built using:
- Python
- TensorFlow / Keras (Xception)
- Flask Web App

---

# 🚀 Quick Start (Run the Project)

## Step 1 — Clone repo
git clone https://github.com/Gaya19/illegal-drug-detection-in-humans-using-dl.git

## Step 2 — Go inside project folder (IMPORTANT)
cd DrugDetectionProject

Make sure you can see:
app.py  
requirements.txt  
templates/  
static/

---

## Step 3 — Create virtual environment
python -m venv venv

## Step 4 — Activate

Windows:
venv\Scripts\activate

---

## Step 5 — Install all dependencies (automatic)
pip install -r requirements.txt

NOTE:
Do NOT manually install tensorflow, numpy, keras.
requirements.txt handles everything automatically.

---

## Step 6 — Run app
python app.py

Open browser:
http://127.0.0.1:5000

---

# 🤖 Model
Pre-trained model included:
xception_model.h5

So training is NOT required for testing.

---

# 🏋️ Training (optional)

If you want to train again:

cd training  
python train_xception.py

---

# 📂 Dataset

Dataset was created by the author.

Due to large size, it is NOT uploaded to GitHub.

Download dataset here:
https://drive.google.com/drive/folders/14bkG6Xq_Y1AE88PTzq4MJMLiVLg8dELR?usp=sharin

After download, extract as:

dataset/
   train/
   test/
   val/

---
## Dataset Setup

⚠️ The dataset is not uploaded to GitHub because it is large.

Download the dataset from Google Drive:
👉 https://drive.google.com/drive/folders/14bkG6Xq_Y1AE88PTzq4MJMLiVLg8dELR?usp=sharin

### Steps

1. Download `dataset.zip`
2. Extract the zip file
3. Move the extracted `dataset` folder into the project root directory


# 📂  Augmented Dataset

An augmented dataset is provided to improve model training.

Download augmented dataset here:
Augmented Dataset
 https://drive.google.com/drive/folders/1-9pVe4R814rqXDE8ZvoHzjCC4sQZStjS?usp=sharing

Steps

Download augmented_dataset.zip

Extract the zip file

Move the extracted folder into the project root directory:

## 💾 Download Pre-trained Models (Single Link)

Both pre-trained models are available in a single folder:

- `xception_model.h5`
- `best_xception_model.h5`

**Download here:**  
https://drive.google.com/drive/folders/1UkvVA1ho0MWnvRJXTPLWBtq4Bzrg-6OL?usp=sharing 

After downloading, place the folder in the project root:


# 📁 Project Structure

DrugDetectionProject/
│
├── app.py  
├── train_xception.py  
├── utils.py  
├── augmentation.py  
├── requirements.txt  
├── README.md  
├── .gitignore  
├── models/
│    ├── xception_model.h5
│    └── best_xception_model.h5
├── templates/  
├── static/  
├── dataset/             # original dataset
└── augmented_dataset/   # optional, augmented images
 

---



# 👩‍💻 Author
Gayathri
