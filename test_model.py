import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("CarPrice_Assignment.csv")
data['brand'] = data['CarName'].apply(lambda x: x.split()[0].lower())

def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Ошибка! Введите число, а не текст.")

def encode(column, user_value):
    le = LabelEncoder()
    all_values = data[column].tolist() + [user_value]
    encoded = le.fit_transform(all_values)
    return encoded[-1]

brand       = input("Бренд (toyota, bmw, honda): ").strip().lower()
engine      = get_number("Объём двигателя (например 130): ")
hp          = get_number("Лошадиные силы (например 111): ")
weight      = get_number("Вес (например 2337): ")
width       = get_number("Ширина (например 64.1): ")
length      = get_number("Длина (например 168.8): ")
cylinders   = input("Цилиндры (four/six/eight): ").strip().lower()
bore        = get_number("Степень сжатия (например 3.47): ")
mpg         = get_number("Расход на шоссе MPG (например 27): ")
stroke      = get_number("Ход поршня (например 2.68): ")
height      = get_number("Высота (например 52.4): ")
drive       = input("Привод (fwd/rwd/4wd): ").strip().lower()
aspiration  = input("Аспирация (std/turbo): ").strip().lower()

brand_num     = encode('brand', brand)
cylinders_num = encode('cylindernumber', cylinders)
drive_num     = encode('drivewheel', drive)
asp_num       = encode('aspiration', aspiration)

car = pd.DataFrame([[engine, hp, weight, width, length,
                      cylinders_num, bore, brand_num,
                      mpg, stroke, height, drive_num, asp_num]],
                   columns=['enginesize', 'horsepower', 'curbweight', 'carwidth',
                            'carlength', 'cylindernumber', 'boreratio', 'brand',
                            'highwaympg', 'stroke', 'carheight', 'drivewheel', 'aspiration'])

price = model.predict(car)[0]
print(f"\n  predicted price: ${price:,.0f}")