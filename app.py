from flask import Flask, render_template, request, redirect, url_for,jsonify,flash,session
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import joblib
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import plotly.express as px
import os
matplotlib.use('Agg')  # Use non-GUI backend


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key

# SQLite Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(150), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Load the trained model for fertilizer recommend
model = joblib.load('fertilizer_app.pkl')

# Load the dataset to fit LabelEncoders
data = pd.read_csv("./data/fertilizer_recommendation.csv")

# Initialize LabelEncoders and fit them with the dataset
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()  # For decoding the prediction
data['Soil Type'] = le_soil.fit_transform(data['Soil Type'])
data['Crop Type'] = le_crop.fit_transform(data['Crop Type'])
data['Fertilizer Name'] = le_fertilizer.fit_transform(data['Fertilizer Name'])


#loading models for yeild prediction
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocesser.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['loggedin'] = True
            session['id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
            return redirect(url_for('login', form='login'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['first-name']
        email = request.form['email']
        password = request.form['password']

        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()

        # Validation logic
        if existing_user:
            if existing_user.username == username:
                flash('Username already exists!', 'error')
            else:
                flash('Email already exists!', 'error')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
        elif not username or not password or not email:
            flash('Please fill out the form!', 'error')
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful! Now login.', 'success')
            return redirect(url_for('login', form='login'))

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/crop-recommend', methods=['GET', 'POST'])
def crop_recommend():
    if request.method == 'POST':
        # Extract form data
        try:
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph_value = float(request.form['phValue'])
            rainfall = float(request.form['rainfall'])
        except ValueError:
            return "Please enter valid numeric values."

        # Input validation
        if not (0 <= nitrogen <= 300):
            return "Nitrogen value must be between 0 and 300 kg/ha."
        if not (0 <= phosphorus <= 150):
            return "Phosphorus value must be between 0 and 150 kg/ha."
        if not (0 <= potassium <= 250):
            return "Potassium value must be between 0 and 250 kg/ha."
        if not (0 <= temperature <= 45):
            return "Temperature must be between 0 and 45 °C."
        if not (0 <= humidity <= 100):
            return "Humidity must be between 0 and 100%."
        if not (4 <= ph_value <= 14):
            return "The pH value must be between 4 and 14."
        if not (0 <= rainfall <= 2000):
            return "Rainfall value must be between 0 and 2000 mm."

        # Create a feature array in the same order as the training data
        values = [nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]
        
        try:
            # Load the model
            model = joblib.load('crop_app.pkl')
            arr = [values]

            # Get probabilities for each class
            probabilities = model.predict_proba(arr)[0]
            # Get the indices of the top 3 crops with the highest probabilities
            top_indices = probabilities.argsort()[-3:][::-1]

            # Assuming you have a list of crop labels corresponding to your model
            crop_labels = model.classes_  # Get the classes from the model
            recommended_crops = crop_labels[top_indices]  # Get the top crop names
            
            # Render the result
            return render_template('crop-recommend.html', prediction=recommended_crops.tolist())
        except Exception as e:
            return f"An error occurred: {e}"
    
    return render_template('crop-recommend.html')


@app.route('/yield-predict', methods=['GET', 'POST'])
def yield_predict():
    # Load unique areas and items from the dataset to populate dropdowns
    # Load your dataset
    df = pd.read_csv('./data/yield_df.csv')
    unique_areas = df['Area'].unique().tolist()
    unique_items = df['Item'].unique().tolist()

    if request.method == 'POST':
        # Extracting form inputs
        Year = int(request.form['Year'])  # Dropdown will always ensure a valid year is selected
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']  # Dropdown will ensure valid area
        Item = request.form['Item']  # Dropdown will ensure valid item

        # Input validation
        if not (1990 <= Year <= 2013):
            return "Year must be between 1990 and 2013."
        if not (51 <= average_rain_fall_mm_per_year <= 3240):
            return "Rainfall must be between 51 and 3240 mm."
        if not (0.04 <= pesticides_tonnes <= 367778):
            return "Pesticides must be between 0.04 and 367,778 tonnes."
        if not (1.3 <= avg_temp <= 40.65):
            return "Average temperature must be between 1.3°C and 40.65°C."

        # Prepare the feature array
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        
        # Feature transformation (using preprocessor)
        transformed_features = preprocessor.transform(features)
        
        # Make the prediction
        prediction = dtr.predict(transformed_features).reshape(1, -1)
        
        # Render the result
        return render_template('yield-predict.html', prediction=prediction[0][0], areas=unique_areas, items=unique_items)
    
    # GET request: render the form with dropdowns populated
    return render_template('yield-predict.html', areas=unique_areas, items=unique_items)


@app.route('/weather-forecast')
def weather_forecast():
    return render_template('weather-forecast.html')

@app.route('/fertilizer-recommend', methods=['GET', 'POST'])
def fertilizer_recommend():
    if request.method == 'POST':
        try:
            # Extract form data
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            soil_moisture = float(request.form['soilMoisture'])
            soil_type = request.form['soilType']
            crop_type = request.form['cropType']
            nitrogen = float(request.form['nitrogen'])
            potassium = float(request.form['potassium'])
            phosphorous = float(request.form['phosphorous'])

            # Encode categorical features
            soil_type_encoded = le_soil.transform([soil_type])[0]
            crop_type_encoded = le_crop.transform([crop_type])[0]

            # Create a feature array
            features = [[temperature, humidity, soil_moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]]

            # Try predicting class directly
            prediction = model.predict(features)
            print("Prediction:", prediction)

            # Decode the prediction back to the original fertilizer name
            recommended_fertilizer = le_fertilizer.inverse_transform(prediction)
            print("Recommended fertilizer:", recommended_fertilizer)

            # Render the result in the template
            return render_template('fertilizer-recommend.html', recommendations=recommended_fertilizer)

        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('fertilizer-recommend.html')



# Load your dataset
df = pd.read_csv('./data/analysis1_data.csv')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    states = df['state'].unique()  # Get unique states from the dataset
    
    if request.method == 'POST':
        state = request.form['state']
        year = request.form['year']
        
        # Filter the data by selected year (for comparison across all crops)
        year_data = df[(df['state'] == state) & (df['year'] == int(year))]

        if year_data.empty:
            return render_template('analysis.html', states=states, error='No data available for the selected year.')

        # Create the first plot: Bar chart for cost of production
        plt.figure(figsize=(10, 6))  # Increased size for better visibility
        plt.bar(year_data['crop_type'], year_data['cost_of_production_per_hectare'], color='blue')
        plt.title(f'Cost of Production for Different Crops in {state} in {year}')
        plt.xlabel('Crop Type')
        plt.ylabel('Cost of Production (per hectare)')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Create the second plot: Pie chart for cultivation area
        # Group by crop_type and sum the cultivation area
        grouped_data = year_data.groupby('crop_type', as_index=False)['cultivation_area_hectares'].sum()

        # Create the second plot: Pie chart for cultivation area
        plt.figure(figsize=(8, 8))  # Increased size for better visibility
        plt.pie(grouped_data['cultivation_area_hectares'], labels=grouped_data['crop_type'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Cultivation Area Distribution in {state} in {year}')
        img2 = io.BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        pie_chart_url = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        # Rainfall effect on crops (filtered by state and year)
        plt.figure(figsize=(10, 6))  # Increased size for better visibility
        plt.bar(year_data['crop_type'], year_data['rainfall_mm'], color='teal')
        plt.title(f'Rainfall Impact on Different Crops in {state} in {year}')
        plt.xlabel('Crop Type')
        plt.ylabel('Rainfall (mm)')
        plt.xticks(rotation=45)
        img3 = io.BytesIO()
        plt.savefig(img3, format='png')
        img3.seek(0)

        # Convert the image to a base64 string
        rainfall_chart_url = base64.b64encode(img3.getvalue()).decode()

        plt.close()

        return render_template('analysis.html', 
                               states=states,
                               graph_url=graph_url, 
                               pie_chart_url=pie_chart_url, 
                               rainfall_chart_url=rainfall_chart_url, 
                               selected_state=state,
                               selected_year=year)

    return render_template('analysis.html', states=states)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'loggedin' in session:
        user_id = session['id']
        user = User.query.get(user_id)

        if request.method == 'POST':
            # Update profile details in the database
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']
            phone = request.form['phone']
            location = request.form['location']

            # Check for uniqueness if username or email changed
            existing_user = User.query.filter(
                ((User.username == username) | (User.email == email)) & (User.id != user_id)
            ).first()

            if existing_user:
                if existing_user.username == username:
                    flash('Username already exists!', 'error')
                else:
                    flash('Email already exists!', 'error')
            else:
                user.username = username
                user.email = email
                user.phone = phone
                user.location = location
                if password:
                    user.set_password(password)
                db.session.commit()
                flash('Profile saved successfully', 'success')
                return redirect(url_for('profile'))

        return render_template('profile.html', account=user)
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    # Here we would normally clear the session or any user-specific data
    return redirect(url_for('index'))


@app.route('/help')
def help_us():
    return render_template('help.html')  # The help/contact page

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
