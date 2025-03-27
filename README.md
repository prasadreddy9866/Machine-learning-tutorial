# 🛒 Superstore Sales Analysis & Profit Prediction

This project analyzes a US Superstore dataset using various data visualization techniques and machine learning models to predict profit based on sales.

---

## 📁 Dataset

 - Download the `US Superstore data.csv` dataset from Repository

---

## ✅ Steps to Run the Project

### 1. Install Python 3.11.4 and Add to PATH

Windows:
- Download from [python.org](https://www.python.org/)
- Check "Add Python to PATH" during installation
- Verify:
```bash
python --version
```

macOS/Linux:
```bash
brew install python@3.11
# or
sudo apt update && sudo apt install python3.11
python3 --version
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/Superstore-ML-Analysis.git
cd Superstore-ML-Analysis
```

---

### 3. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### 5. Run the Analysis Script

```bash
python SUPER_STORE_ANALYSIS.py
```

You will see:
- Visualizations of sales/profit/category
- Multiple ML model evaluations including Linear Regression, Decision Tree, Random Forest, KNN, and SVM

---

## 📊 Output

- Bar charts, pie charts, scatter plots
- Model performance comparison based on MSE

---

