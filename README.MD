# SecureBot — Predictive Maintenance ML App

A Streamlit web application for predicting machine failures using the **AI4I 2020 Predictive Maintenance Dataset**. The app trains four classifiers (KNN, Decision Tree, Random Forest, SVM) under two pipelines (raw imbalanced data vs. class-balanced data), evaluates them side-by-side, and provides a live prediction dashboard for operators.

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Installation Guide](#-installation-guide)
   - [Standard Installation](#standard-installation-maclinux)
   - [Manual Installation (Troubleshooting)](#manual-installation-troubleshooting)
4. [Running the App](#-running-the-app)
5. [Using the App](#-using-the-app)
6. [Test Cases](#-test-cases)
7. [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

**Domain:** Predictive Maintenance (Industry 4.0 / IoT)

In a factory setting, unexpected machine failures cost money through downtime, broken parts, and rushed repairs. This project predicts whether a machine will fail in the near future based on 5 sensor readings (air temperature, process temperature, rotational speed, torque, tool wear) and the machine's quality type (Low/Medium/High).

### The Machine Learning Challenge: Class Imbalance

Out of 10,000 machine records, only ~339 (3.4%) actually failed. If a model simply guesses "No Failure" every time, it achieves 96.6% accuracy but is completely useless in practice because it never catches actual failures.

To solve this, the app implements **two pipelines**:
1. **Before Balancing:** Trains on the raw, highly imbalanced data (baseline).
2. **After Balancing:** Uses `RandomOverSampler` + `RandomUnderSampler` to create a 50/50 training set, allowing the models to actually learn what a failure looks like.

### Algorithms Compared
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest** (ensemble of 100 decision trees)
- **Support Vector Machine (SVM)**

---

## 📂 Project Structure

```
securebot_app/
├── app.py                     # Streamlit entry point
├── requirements.txt           # Python dependencies
├── run.sh                     # Launcher for macOS / Linux
├── run.bat                    # Launcher for Windows
├── data/
│   └── ai4i2020.csv           # Dataset (10,000 rows)
└── src/
    ├── __init__.py
    ├── data_loader.py         # Dataset loading & cleaning
    ├── preprocessing.py       # Encoding, scaling, train/test split, balancing
    ├── models.py              # Model training & evaluation
    ├── visualize.py           # Matplotlib/Seaborn plotting helpers
    └── predict.py             # Live single-row inference
```

---

## 🚀 Installation Guide

### Prerequisites
- **Python 3.9 to 3.12** (⚠️ Python 3.13 is NOT recommended as many data science libraries lack pre-built wheels for it yet).
- `pip` package manager.

### Standard Installation (Mac/Linux)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/securebot_app.git
   cd securebot_app
   ```

2. **Make the launcher executable:**
   ```bash
   chmod +x run.sh
   ```

3. **Run the launcher:**
   ```bash
   ./run.sh
   ```
   *The first run will automatically create a virtual environment (`venv/`), install all dependencies, and launch Streamlit. This takes about 60 seconds. Subsequent runs are instant.*

### Manual Installation (Troubleshooting)

If the `run.sh` script fails (common on Arch Linux, bleeding-edge distros, or systems where `python` points to Python 3.13), follow these steps:

1. **Ensure you have Python 3.12 installed.**
   - *Arch Linux:* Install from AUR: `yay -S python312`
   - *Ubuntu/Debian:* `sudo apt install python3.12 python3.12-venv`
   - *Mac:* `brew install python@3.12`

2. **Create the virtual environment manually using Python 3.12:**
   ```bash
   rm -rf venv
   python3.12 -m venv venv
   ```

3. **Upgrade pip:**
   ```bash
   ./venv/bin/python -m pip install --upgrade pip
   ```

4. **Install dependencies:**
   If `pip install -r requirements.txt` fails, install packages individually:
   ```bash
   ./venv/bin/pip install streamlit
   ./venv/bin/pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

5. **Launch Streamlit directly:**
   ```bash
   ./venv/bin/streamlit run app.py
   ```

### Windows Installation

1. **Clone the repository** and navigate into it.
2. Ensure Python 3.9–3.12 is installed (from [python.org](https://www.python.org/downloads/)).
3. Double-click `run.bat`, or run it from Command Prompt:
   ```bat
   run.bat
   ```

---

## ▶️ Running the App

Once the setup is complete, the app will launch automatically and print a local URL:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open `http://localhost:8501` in your web browser. Use the sidebar on the left to navigate between the Overview, EDA, Model Comparison, and Live Prediction tabs.

---

## 🖥️ Using the App

The app has 5 tabs accessible from the sidebar:

1. **📊 Overview:** Dataset statistics, class distribution, and statistical summaries.
2. **🔍 EDA (Exploratory Data Analysis):** Visualizations of feature distributions, boxplots comparing failures vs non-failures, and a correlation heatmap.
3. **🤖 Models (Before):** Baseline performance of all 4 classifiers on imbalanced data. High accuracy but poor recall.
4. **⚖️ Models (After):** Performance after oversampling/undersampling. Shows dramatic recall improvement and a before-vs-after comparison chart.
5. **🎯 Live Prediction:** An interactive dashboard where you input sensor values and the model predicts failure probability in real-time.

---

## 🧪 Test Cases

Use these scenarios in the **Live Prediction** tab to verify the model works. Select the **Balanced pipeline** and **Random Forest** model for all tests.

| Test Case | Air Temp [K] | Process Temp [K] | Rot. Speed [rpm] | Torque [Nm] | Tool Wear [min] | Type | Expected Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Normal Operation** | 298.0 | 308.6 | 1551 | 42.8 | 0 | M | 🟢 No Failure |
| **2. Tool Wear Failure (TWF)** | 298.0 | 308.6 | 1551 | 42.8 | 215 | M | 🔴 Failure |
| **3. Heat Dissipation (HDF)** | 300.0 | 309.0 | 1300 | 50.0 | 50 | L | 🔴 Failure |
| **4. Overstrain (OSF)** | 298.0 | 308.6 | 1400 | 60.0 | 200 | L | 🔴 Failure |
| **5. Edge Case (Safe Power)** | 295.0 | 305.0 | 1900 | 35.0 | 80 | H | 🟢 No Failure |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
