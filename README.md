# Lending Club Prediction App

The Model is deployed into a Streamlit App and can be access here: 
<https://lendingclubiti105project.streamlit.app>

## Contributors

Name: **Brian Sum**  
Email: <sumkh2@gmail.com>

## Repository Structure

```bash
lending_club/
│
├── src/                            # Contains all Python scripts and artifacts
│   ├── best_model.joblib           # Trained model for the application
│   ├── vec_preprocessor.joblib.py  # Fitted Vectorizer for data preprocessing
│   ├── util.py                     # Utility functions
│   └── run.py                      # Main script running the application
│
├── datasets/                       # Contains datasets
│   ├── lc_datasets.csv             # Lending Club dataset
│
├── EDA_Report.ipynb      # Jupyter notebook for Exploratory Data Analysis
├── ML_Report.ipynb       # Jupyter notebook for Machine Learning
├── requirements.txt      # Python dependencies for the streamlit application
└── README.md             # Project documentation and setup instructions

```

## Setup and Execution

### Prerequisites

- Python 3.9+ installed
- pip (Python package installer)

### Environment Setup

To set up and activate a virtual environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sumkh/Lending_Club_ITI105_Project.git
   cd Lending_Club_ITI105_Project
   ```

   Use `cd` to move into the project directory where you want to create your environment (e.g., cd my_project)

2. **Create a virtual environment:**

   ```bash
   python3 -m venv yourenv
   ```

   Replace "env" with the name you want for your environment, like `yourenv`. This creates a folder named `yourenv` (or your chosen name) in your project directory.

3. **Activate the virtual environment:**

   ```bash
   source yourenv/bin/activate
   ```

   Replace `yourenv` if you used a different name.

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Execute the streamlit app on local device with the following command:

```bash
streamlit run app.py
```

### Running MLFlow on Local Devices

1. **Install dependencies:**

   ```bash
   pip install mlflow
   ```

2. **Execute the following command**

   Ensure the sqlite database file `mlflow.db` is in the folder directory before executing the following command. Otherwise, the mlflow server will generate a new one. (Overwrite the neewly generated `mlflow.db` if necessary)

   ```bash
   mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlruns \
      --host 127.0.0.1 \
      --port 8080
   ```
