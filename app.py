from flask import Flask, render_template, request, jsonify 
import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
data = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global data
    file = request.files['file']
    content = file.stream.read().decode('utf-8')
    data = pd.read_csv(StringIO(content))
    data = data[data['Year'] <= 2025]
    return jsonify({'cities': sorted(data['City'].unique().tolist())})

@app.route('/data')
def get_data():
    cities = request.args.getlist('cities[]')
    if not cities:
        return jsonify({"error": "No cities selected"}), 400

    df = data[data['City'].isin(cities)]
    all_rows = []

    for city in cities:
        df_city = df[df['City'] == city]
        for col in ['PM2.5', 'NO2', 'CO2', 'GreenCover']:
            model = LinearRegression()
            X = df_city['Year'].values.reshape(-1, 1)
            y = df_city[col].values
            if len(y) < 2:
                continue
            model.fit(X, y)
            for year in [2026, 2027, 2028]:
                pred = model.predict(np.array([[year]]))[0]
                all_rows.append({
                    'City': city,
                    'Year': year,
                    col: pred
                })

    forecast_df = pd.DataFrame(all_rows).groupby(['City', 'Year']).first().reset_index()
    df = pd.concat([df, forecast_df], ignore_index=True).sort_values(by=['City', 'Year'])

    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
