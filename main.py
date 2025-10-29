from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset and model
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    # Prepare dropdown values
    bedrooms = sorted(map(str, data['beds'].unique()))
    bathrooms = sorted(map(str, data['baths'].unique()))
    sizes = sorted(map(str, data['size'].unique()))
    zip_codes = sorted(map(str, data['zip_code'].unique()))

    return render_template(
        'index.html',
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        sizes=sizes,
        zip_codes=zip_codes
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Validate inputs
        if not all([bedrooms, bathrooms, size, zipcode]):
            return jsonify({'error': 'Missing input values'}), 400

        # Prepare DataFrame
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                                  columns=['beds', 'baths', 'size', 'zip_code'])

        # Convert datatypes safely
        for col in ['beds', 'baths', 'size', 'zip_code']:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
            if input_data[col].isnull().any():
                input_data[col] = data[col].mode()[0]  # Replace with mode

        # Handle unknown values
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                input_data[column] = data[column].mode()[0]

        # Predict
        prediction = pipe.predict(input_data)[0]
        price = round(float(prediction), 2)

        # Return JSON result
        return jsonify({'price': price})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}),500

if __name__=="__main__":
    app.run(debug=True, port=5000)

