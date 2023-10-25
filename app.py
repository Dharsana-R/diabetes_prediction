import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Handle missing values by replacing 0 with NaN
diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Drop rows with missing values
diabetes_df.dropna(inplace=True)

# Split the dataset into features (X) and target (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Standardize the data
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Create and train the Random Forest model
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X, y)

# Save the trained model to a pickle file
with open('diabetes_rfc_model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)

# Home page with a form for input
@app.route('/')
def home():
    return render_template('index.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        input_data = [float(request.form['Pregnancies']), float(request.form['Glucose']), float(request.form['BloodPressure']),
                      float(request.form['SkinThickness']), float(request.form['Insulin']), float(request.form['BMI']),
                      float(request.form['DiabetesPedigreeFunction']), float(request.form['Age'])]


        # Standardize the input data
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        std_input_data = sc_X.transform(input_data_as_numpy_array)


        # Make a prediction using the trained classifier
        prediction = rfc.predict(std_input_data)


        result = 'The person is diabetic.' if prediction[0] == 1 else 'The person is not diabetic.'


        return render_template('result.html', prediction=result)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
