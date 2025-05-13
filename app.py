
from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure static directory exists for saving plots
if not os.path.exists('static'):
    os.makedirs('static')

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

# ------------------- Visualization Generation -------------------
# Generate ROC Curve
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {np.trapz(tpr, fpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('static/roc_curve.png')
plt.close()

# Generate Feature Importance Plot
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(6, 4))
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
plt.xticks(range(len(feature_importance)), np.array(feature_names)[sorted_idx], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# Generate Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('static/confusion_matrix.png')
plt.close()

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
        .navbar {
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
            display: flex;
            align-items: center;
        }
        .navbar h1 {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            gap: 2rem;
        }
        .section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h2 {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
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
        .output-section p {
            color: #d1fae5;
            font-size: 1.1rem;
            text-align: center;
        }
        .visualization-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            padding: 1rem 0;
        }
        .visualization-images {
            display: flex;
            flex-direction: row;
            gap: 1.5rem;
            overflow-x: auto;
        }
        .visualization-images img {
            max-width: 400px;
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        @media (max-width: 900px) {
            .visualization-images {
                flex-direction: column;
                overflow-x: visible;
            }
            .visualization-images img {
                max-width: 100%;
            }
        }
        @media (max-width: 600px) {
            .navbar {
                padding: 0.8rem 1.5rem;
            }
            .navbar h1 {
                font-size: 1.4rem;
            }
            .section {
                padding: 1.5rem;
            }
            h2 {
                font-size: 1.5rem;
            }
            form {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>
    <nav class="navbar">
        <h1>Customer Churn Prediction</h1>
    </nav>
    <div class="container">
        <div class="section input-section">
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
        <div class="section output-section">
            <h2>Prediction Result</h2>
            {% if result %}
                <p>{{ result }}</p>
            {% else %}
                <p>No prediction yet. Submit the form to see the result.</p>
            {% endif %}
        </div>
        <div class="section visualization-section">
            <h2>Model Visualizations</h2>
            <div class="visualization-images">
                <img src="{{ url_for('static', filename='roc_curve.png') }}" alt="ROC Curve">
                <img src="{{ url_for('static', filename='feature_importance.png') }}" alt="Feature Importance">
                <img src="{{ url_for('static', filename='confusion_matrix.png') }}" alt="Confusion Matrix">
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="{{ url_for('static', filename='js/particles.min.js') }}"></script>
    <script src="{{ url_for('static', filename='js/particles-config.js') }}"></script>
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
