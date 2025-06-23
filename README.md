Heart Disease Prediction App
This is a web-based application built with Streamlit to predict the likelihood of heart disease based on patient health data. The app uses a machine learning model trained on the UCI Heart Disease dataset and provides personalized health recommendations. It features a responsive design, making it accessible on desktops, tablets, and smartphones.
Features

Prediction: Enter patient details (e.g., age, cholesterol, blood pressure) to predict heart disease risk.
Health Recommendations: Receive diet, exercise, and lifestyle advice based on prediction results.
Feature Insights: Visualize the importance of each feature in the prediction model.
Responsive UI: Optimized for all devices with touch-friendly inputs and scalable charts.
Error Handling: Validates inputs and checks for model files.

Prerequisites
Python 3.8+
Virtual environment (recommended)
Git (for cloning the repository)
Streamlit Cloud account (for deployment)

Installation
Clone the Repository:
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction


Set Up Virtual Environment:
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux


Install Dependencies:
pip install -r requirements.txt


Ensure Model Files:
The repository includes heart_disease_model.pkl and scaler.pkl. If missing, generate them:python heart_disease_prediction.py
Requires heart.csv (UCI Heart Disease dataset). Download from Kaggle or UCI and place in the project directory.



Running Locally
Activate the virtual environment:venv\Scripts\activate  # Windows


Run the Streamlit app:streamlit run app.py
Open http://localhost:8501 in a browser.
Test on other devices on the same network using your computer’s IP (e.g., http://192.168.x.x:8501). Find your IP:ipconfig  # Windows



Deployment to Streamlit Cloud
Push to GitHub:
git add .
git commit -m "Add README and app files"
git push origin main


Deploy:
Log in to Streamlit Cloud.
Create a new app, select your repository, and set app.py as the main file.
Deploy and access the app via the provided URL  https://your-app-name.streamlit.app)](https://heart-diseases-predictor-a8dlcsr8idu8xwtzkfxcbg.streamlit.app/.

Test on Devices:
Open the URL on desktops, tablets, and smartphones to verify responsiveness.

Usage

Prediction Tab:
Enter patient details (e.g., age, sex, cholesterol, blood pressure).
Click Predict Risk to see the heart disease probability and recommendations.
Recommendations cover diet, exercise, and lifestyle based on input values and risk level.


Feature Insights Tab:
View a bar chart showing the importance of each feature in the prediction model.


Input Validation:

Ensure cholesterol and blood pressure are positive.
All inputs have default values for ease of use.



Project Structure

app.py: Main Streamlit app file.
requirements.txt: Python dependencies.
heart_disease_model.pkl: Trained RandomForest model.
scaler.pkl: StandardScaler for numerical features.
heart_disease_prediction.py: Script to train the model and generate model files.
README.md: Project documentation.

Dataset
The model is trained on the UCI Heart Disease dataset, containing features like age, sex, chest pain type, cholesterol, and ST slope. The dataset (heart.csv) is required to retrain the model.
Notes
Jupyter Users: Avoid running app.py in Jupyter notebooks to prevent kernel conflicts. Use streamlit run app.py from the command line.
Virtual Environment: Use the provided virtual environment (venv) to avoid dependency conflicts.
Cross-Device Access: The app is optimized for all screen sizes with responsive CSS and touch-friendly inputs.

