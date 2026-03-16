from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

app = Flask(__name__)

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("CarPrice_Assignment.csv")
data['brand'] = data['CarName'].apply(lambda x: x.split()[0].lower())

df = data.copy()
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df[['enginesize','horsepower','curbweight','carwidth','carlength',
        'cylindernumber','boreratio','brand','highwaympg','stroke',
        'carheight','drivewheel','aspiration']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_pred = model.predict(X_test)

r2   = round(r2_score(y_test, y_pred), 4)
mae  = round(mean_absolute_error(y_test, y_pred), 2)
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)

scatter_actual    = [round(float(v), 0) for v in y_test.tolist()]
scatter_predicted = [round(float(v), 0) for v in y_pred.tolist()]

sample_cols = ['brand','enginesize','horsepower','curbweight','carwidth',
               'carlength','cylindernumber','boreratio','highwaympg',
               'stroke','carheight','drivewheel','aspiration','price']
samples = data[sample_cols].head(10).to_dict('records')

def encode(column, user_value):
    le2 = LabelEncoder()
    all_values = data[column].tolist() + [user_value]
    encoded = le2.fit_transform(all_values)
    return int(encoded[-1])

@app.route("/")
def index():
    brands = sorted(data['brand'].unique().tolist())
    return render_template("index.html",
        brands=brands,
        r2=r2, mae=mae, rmse=rmse,
        scatter_actual=scatter_actual,
        scatter_predicted=scatter_predicted,
        samples=samples
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        brand_num     = encode('brand',          d['brand'])
        cylinders_num = encode('cylindernumber', d['cylinders'])
        drive_num     = encode('drivewheel',     d['drive'])
        asp_num       = encode('aspiration',     d['aspiration'])

        car = pd.DataFrame([[
            float(d['enginesize']), float(d['horsepower']),
            float(d['curbweight']), float(d['carwidth']),
            float(d['carlength']),  cylinders_num,
            float(d['boreratio']),  brand_num,
            float(d['highwaympg']), float(d['stroke']),
            float(d['carheight']),  drive_num, asp_num
        ]], columns=['enginesize','horsepower','curbweight','carwidth',
                     'carlength','cylindernumber','boreratio','brand',
                     'highwaympg','stroke','carheight','drivewheel','aspiration'])

        price = model.predict(car)[0]
        return jsonify({"price": round(float(price), 2), "status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)