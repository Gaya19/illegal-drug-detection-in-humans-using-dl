# Illicit Drug Detection (Xception)

A Flask web app to detect illicit drug influence from **eye images** using the **Xception** CNN model.

## Project Structure
```
DrugDetectionProject/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── train/
│   │   ├── Normal/
│   │   └── Drug/
│   └── test/
│       ├── Normal/
│       └── Drug/
├── models/
│   └── xception_model.h5            # Put your trained model here
├── static/
│   ├── css/style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── about.html
│   └── result.html
└── training/
    ├── train_xception.py
    └── utils.py
```

## Quick Start
1. Create a venv and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your trained model at `models/xception_model.h5` (see `training/train_xception.py` to train).
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Training
- Put your images into `dataset/train/Normal`, `dataset/train/Drug`.
- (Optional) Move a portion to `dataset/test/...` for held-out testing.
- Run:
  ```bash
  python training/train_xception.py
  ```
- The best model will be saved to `models/xception_model.h5`.

## Notes
- Input size: **299x299** (Xception), images scaled to **[0,1]**.
- You can change class labels in `app.py` (`labels = ["Normal", "Drug Affected"]`).

## Credits
- UI built to match your provided screenshot (navbar, hero, footer).
