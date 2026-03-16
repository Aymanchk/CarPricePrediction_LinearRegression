# Car Price Predictor

A machine learning web application that predicts car prices using Linear Regression.

---

## About

This project uses the **CarPrice Assignment dataset** to train a Linear Regression model that predicts the price of a car based on its technical specifications. The model achieves **96.3% R² accuracy** on the test set.

The web interface is built with **Flask + HTML/CSS** and allows users to input car parameters and get an instant price prediction.

---

## Model

- **Algorithm:** Linear Regression (scikit-learn)
- **Accuracy:** R² = 0.963
- **Features used (13):**
  - `enginesize`, `horsepower`, `curbweight`, `carwidth`, `carlength`
  - `cylindernumber`, `boreratio`, `brand`, `highwaympg`
  - `stroke`, `carheight`, `drivewheel`, `aspiration`

- **Removed columns:** `CarName` was replaced with `brand` (first word only). Columns like `fueltype`, `enginetype`, `fuelsystem`, `doornumber`, `wheelbase`, `citympg` were excluded due to low correlation with price or redundancy.

---

## Project Structure

```
car-price-predictor/
├── app.py                   # Flask server + model loading + /predict endpoint
├── car_price_model.pkl      # Trained Linear Regression model
├── CarPrice_Assignment.csv  # Dataset
├── model.py                 # Model training script
├── templates/
│   └── index.html           # Web interface (Stage 3)
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install flask pandas scikit-learn numpy
```

**2. Train the model (if needed)**
```bash
python model.py
```

**3. Start the Flask server**
```bash
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5000
```

---

## Web Interface

- Displays **model accuracy** (R² score)
- Shows the **predicted vs actual price scatter plot**
- Displays **10 sample rows** from the dataset with preprocessing notes
- **Input form** with all 13 features
- Clicking **"Предсказать цену"** returns the predicted price instantly

---

## Dataset

- **Source:** CarPrice Assignment (UCI / Kaggle)
- **Rows:** 205 cars
- **Target variable:** `price` (USD)
- **Train/Test split:** 90% / 10%

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| ML | scikit-learn, pandas, numpy |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript, Chart.js |

---

## Author

Made as part of a Machine Learning course project (Stage 1–3).
