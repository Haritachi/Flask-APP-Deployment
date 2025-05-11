from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os

# Initialize Flask app
app = Flask(__name__)

# ------------------- Model Training -------------------
df = pd.read_csv("Churn2.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

object_columns = df.select_dtypes(include="object").columns
encoders = {}
choices = {}
for column in object_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder
    choices[column] = list(encoder.classes_)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

rfc = RandomForestClassifier(random_state=42)
dtc = DecisionTreeClassifier(random_state=42)

rfc.fit(X_train, y_train)
dtc.fit(X_train, y_train)

rfc_acc = accuracy_score(y_test, rfc.predict(X_test))
dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

best_model = rfc if rfc_acc >= dtc_acc else dtc
best_model_name = "Random Forest" if rfc_acc >= dtc_acc else "Decision Tree"

feature_names = X.columns.tolist()

# ------------------- HTML Template -------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Customer Churn Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%);
            font-family: 'Inter', sans-serif;
            padding: 20px;
        }
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.7;
        }
        .title-bar {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            padding: 1rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: sticky;
            top: 20px;
            z-index: 1000;
            text-align: center;
        }
        .title-bar h1 {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2.5rem;
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h2 {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 2rem;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        .form-group {
            position: relative;
        }
        label {
            color: #e0e0e0;
            font-size: 0.9rem;
            font-weight: 500;
            display: block;
            margin-bottom: 0.5rem;
            transition: color 0.3s ease;
        }
        select, input[type="text"] {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #06b6d4;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        select option {
            background: #1e1e2f;
            color: #06b6d4;
        }
        select:focus, input[type="text"]:focus {
            outline: none;
            border-color: #3b82f6;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        select:hover, input[type="text"]:hover {
            border-color: #60a5fa;
        }
        input[type="submit"] {
            grid-column: 1 / -1;
            padding: 14px;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        input[type="submit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 2000;
            justify-content: center;
            align-items: center;
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            max-width: 500px;
            width: 90%;
            color: #d1fae5;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            animation: slideIn 0.3s ease;
            position: relative;
            text-align: center;
        }
        .modal-content p {
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .close-btn {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: none;
            border: none;
            color: #e0e0e0;
            font-size: 1.5rem;
            cursor: pointer;
            transition: color 0.3s ease;
        }
        .close-btn:hover {
            color: #3b82f6;
        }
        .modal-btn {
            padding: 10px 20px;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .modal-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @media (max-width: 600px) {
            .title-bar {
                padding: 0.8rem 1.5rem;
            }
            .title-bar h1 {
                font-size: 1.4rem;
            }
            .container {
                padding: 1.5rem;
            }
            h2 {
                font-size: 1.5rem;
            }
            form {
                grid-template-columns: 1fr;
            }
            .modal-content {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>
    <div class="title-bar">
        <h1>Customer Churn Prediction</h1>
    </div>
    <div class="container">
        <h2>Enter Customer Details</h2>
        <form method="POST">
            {% for feature in form_fields %}
                <div class="form-group">
                    <label>{{ feature }}:</label>
                    {% if feature in dropdowns %}
                        <select name="{{ feature }}">
                            {% for option in dropdowns[feature] %}
                                <option value="{{ option }}">{{ option }}</option>
                            {% endfor %}
                        </select>
                    {% else %}
                        <input type="text" name="{{ feature }}" required>
                    {% endif %}
                </div>
            {% endfor %}
            <input type="submit" value="Predict">
        </form>
    </div>
    {% if result %}
        <div id="predictionModal" class="modal active">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal()">×</button>
                <p>{{ result }}</p>
                <button class="modal-btn" onclick="closeModal()">OK</button>
            </div>
        </div>
    {% endif %}

    <!-- Scripts -->
    <script src="{{ url_for('static', filename='js/particles.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/particles-config.js') }}"></script>
    <script>
        function closeModal() {
            document.getElementById('predictionModal').classList.remove('active');
        }
    </script>
</body>
</html>
"""

# ------------------- Flask Route -------------------
@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            input_data = {}
            for feature in feature_names:
                value = request.form[feature]
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    input_data[feature] = value

            input_df = pd.DataFrame([input_data])

            for col, encoder in encoders.items():
                input_df[col] = encoder.transform(input_df[col])

            pred = best_model.predict(input_df)[0]
            prob = best_model.predict_proba(input_df)[0][pred]
            result = f"{best_model_name} Prediction: {'Churn' if pred == 1 else 'No Churn'} (Probability: {prob:.2f})"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, form_fields=feature_names, result=result, dropdowns=choices)

# ------------------- Main Entry -------------------
if __name__ == '__main__':
    app.run(debug=True)
    import os
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port, debug=False)
