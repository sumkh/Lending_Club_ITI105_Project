# Lending Club Prediction App

## Contributors

Name: **[Brian Sum]**  
Email: <sumkh2@gmail.com>

## Repository Structure

```bash
solar-panel-efficiency/
│
├── src/                            # Contains all Python scripts and artifacts
│   ├── best_model.joblib           # Trained model for the application
│   ├── vec_preprocessor.joblib.py  # Fitted Vectorizer for data preprocessing
│   ├── util.py                     # Utility functions
│   └── run.py                      # Main script running the application
│
├── requirements.txt      # Python dependencies for the application
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

Execute the pipeline with the following command:

```bash
streamlit run app.py
```
