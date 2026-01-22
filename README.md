# ActivitySense - Smart Human Activity Recognition 🏃

A machine learning application that uses smartphone sensor data to automatically detect and classify human activities. Built with Streamlit for an interactive web interface.

## What Does This Do?

Ever wondered how your fitness tracker knows whether you're walking, running, or sitting? This project demonstrates exactly that! Using data from smartphone accelerometers and gyroscopes, the system can recognize six different activities:

- 🚶 **WALKING** - Normal walking on flat ground
- ⬆️ **WALKING_UPSTAIRS** - Climbing stairs
- ⬇️ **WALKING_DOWNSTAIRS** - Descending stairs
- 🪑 **SITTING** - Seated position
- 🧍 **STANDING** - Standing still
- 🛌 **LAYING** - Lying down

## Features

### 📊 Visualization
- Interactive scatter plots showing activity patterns
- Confusion matrix to see model performance
- Distribution analysis of different activities
- Missing data visualization and handling

### 🔮 Interactive Prediction
- Predict activities using custom sensor inputs
- Pre-loaded examples for each activity type
- Real-time probability scores
- Detailed component explanations

### ⭐ Feature Importance
- See which sensor measurements matter most
- Principal Component Analysis (PCA) explained
- Top 20 most important features visualized

## How It Works

1. **Data Collection**: The system uses 561 sensor measurements from smartphones
2. **Data Cleaning**: Handles missing values and normalizes the data
3. **Dimensionality Reduction**: PCA reduces complexity from 561 to 20 key components
4. **Classification**: Random Forest algorithm predicts the activity with high accuracy

## Getting Started

### Prerequisites

Make sure you have Python installed (3.7 or higher recommended).

### Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```

### Running the App

Simply run:
```bash
streamlit run har_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The project uses `train.csv` which contains sensor data collected from smartphones. The dataset includes measurements from:
- 3-axial linear acceleration (accelerometer)
- 3-axial angular velocity (gyroscope)
- Various derived features like jerk, magnitude, and frequency domain signals

**Dataset Source**: [Human Activity Recognition Using Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) on Kaggle

## Project Structure

```
.
├── har_app.py       # Main Streamlit application
├── train.csv        # Training dataset with sensor readings
└── README.md        # This file
```

## Technical Details

- **Model**: Random Forest Classifier (100 estimators)
- **Accuracy**: Displayed on the Home page after data loading
- **Data Split**: 70% training, 30% testing
- **PCA Components**: 20 principal components retained
- **Missing Values**: Handled by mean imputation

## Navigate the App

- **🏠 Home**: Overview and dataset statistics
- **📊 Visualization**: Explore data patterns and model performance
- **🔮 Interactive Prediction**: Try the model with custom inputs
- **⭐ Feature Importance**: Understand what drives predictions

## Future Improvements

- Real-time sensor data integration
- Additional activities (running, cycling, etc.)
- Model comparison (SVM, Neural Networks)
- Export predictions to CSV
- Mobile app integration

## Built With

- **Streamlit** - Web interface
- **Scikit-learn** - Machine learning models
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualizations

## Notes

This is an educational project demonstrating machine learning concepts in activity recognition. The model is trained on a specific dataset and may need retraining for different sensor configurations or user demographics.

---

*Created for Big Data course, Semester 5*
