# Parkinsons_Detection

An interactive Python project to detect Parkinson’s disease using biomedical voice data. The project automatically downloads the latest dataset from Kaggle, preprocesses it, trains a Random Forest model, and evaluates its performance with metrics and visualization.

---

## Features

- Auto-downloads the latest Parkinson’s dataset from Kaggle.
- Preprocesses the dataset (handles missing values, scales features).
- Trains a Random Forest Classifier.
- Evaluates the model with accuracy, classification report, and confusion matrix.
- Saves a confusion matrix plot in the `visualizations` folder.
- Easy-to-run Python script.

---

## Folder Structure

Parkinsons_Detection/
│
├── data/ # Folder for CSV dataset
│ └── parkinsons.csv
├── visualizations/ # Folder for plots
│ └── plots.png
├── parkinsons_detection.py # Main Python script
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Installation & Setup

1. **Clone the repository:**
    ```bash
git clone https://github.com/Rohith723/Parkinsons_Detection.git
cd Parkinsons_Detection
2. **Install dependencies:**
   pip install -r requirements.txt
3. **Run Script:**
    python parkinsons_detection.py

