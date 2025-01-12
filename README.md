# 📊 Soft Margin SVM Implementation

**Author:** Eshan Jain  

---

## 🚀 Project Overview

This repository contains the implementation of a **Soft Margin Support Vector Machine (SVM)** for binary classification, with support for custom kernel functions and parameter tuning. The implementation also includes kernel performance analysis and hyperparameter tuning on provided datasets.

---

## 🛠️ Features

1. **Kernel Functionality**: 
   - Polynomial Kernel: Customizable with degree, coefficient, and gamma parameters.
   - RBF Kernel: Tunable gamma parameter for performance optimization.
   - Linear Kernel: For baseline performance evaluation.

2. **Optimization**:
   - Solves the dual optimization problem using a **Quadratic Programming Solver**.
   - Learns parameters for the SVM model and supports prediction.

3. **Performance Metrics**:
   - **Accuracy** and **F1 Score** are computed for model evaluation across kernel and parameter settings.

---

## 📈 Experimental Results

### 1️⃣ Linear Kernel:
| **C Value** | **Accuracy (%)** | **F1 Score** |
|-------------|------------------|--------------|
| 0.001       | 92.31            | 0.91428      |
| 0.01        | 92.31            | 0.91428      |
| 0.1         | 92.31            | 0.91428      |
| 1           | 92.31            | 0.91428      |
| 10          | 92.31            | 0.91428      |

### 2️⃣ RBF Kernel (Gamma = 0.1, 0.01, 0.001):
| **C Value** | **Accuracy (%)** | **F1 Score** |
|-------------|------------------|--------------|
| 0.01        | 87.18            | 0.83870      |
| 0.1         | 89.74            | 0.88880      |
| 1           | 89.74            | 0.88235      |
| 10          | 92.31            | 0.91420      |

---

## 🔍 Kernel Analysis

### Linear Kernel:
- Provides consistent accuracy and F1 scores across all **C values**.  
- Ideal for linearly separable data.

### RBF Kernel:
- Shows variability based on **gamma** and **C values**.  
- Higher accuracy observed for **C = 10** and **gamma = 0.1**.

---

## 🧰 Installation and Usage

### Prerequisites:
- Python 3.x
- Libraries: `numpy`, `scipy`, `matplotlib` (for optional visualizations)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/eshan-292/soft-margin-svm.git
   cd soft-margin-svm
   
2.	Run the SVM implementation:
   ```bash
   python svm.py --kernel <kernel_type> --C <value> --gamma <value>

Replace <kernel_type> with linear, polynomial, or rbf.
  
3.  📚 Directory Structure
soft-margin-svm/
│
├── svm.py                 # Main SVM implementation
├── kernels.py             # Kernel functions
├── utils.py               # Helper functions
├── data/                  # Dataset files
├── results/               # Output results and logs
└── README.md              # Project description


