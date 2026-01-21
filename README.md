<div align="center">

# 🚢 Titanic Survival Prediction System
<b>Project 4 — Machine Learning Web GUI (Flask)</b>

<p>
A web-based application that predicts whether a passenger survived the Titanic disaster using a trained Machine Learning model.
</p>

<p>
<a href="#-live-demo">Live Demo</a> •
<a href="#-features">Features</a> •
<a href="#-tech-stack">Tech Stack</a> •
<a href="#-project-structure">Project Structure</a> •
<a href="#-how-to-run-locally">Run Locally</a> •
<a href="#-deployment-render">Deploy (Render)</a>
</p>

<hr style="width:80%;"/>

</div>

---

## 🧠 Project Overview

This project trains a Machine Learning classifier using the **Titanic: Machine Learning from Disaster** dataset and provides a simple **Web GUI** that allows users to enter passenger details and get a prediction:

✅ **Survived**  
❌ **Did Not Survive**

---

## ✅ Features

<table>
  <tr>
    <td width="50%">

### Model Development
- Loads Titanic dataset (`titanic.csv`)
- Handles missing values  
- Encodes categorical variables
- Trains a classifier (Logistic Regression)
- Evaluates using a **classification report**
- Saves the model to disk (`.pkl`)
- Reloads saved model and predicts without retraining

    </td>
    <td width="50%">

### Web GUI (Flask)
- Simple HTML form for passenger input
- Loads trained model and predicts survival
- Displays result instantly in the browser
- Render-ready deployment (binds to `PORT`)

    </td>
  </tr>
</table>

---

## 🧾 Selected Features (Inputs)

The predictive model uses **five (5)** input features:

- **Pclass**
- **Sex**
- **Age**
- **Fare**
- **Embarked**

Target variable:
- **Survived**

---

## 🧰 Tech Stack

<div>

- **Python**
- **Flask** (Web GUI)
- **Pandas / NumPy** (Data handling)
- **Scikit-learn** (Model training)
- **Joblib** (Model persistence)
- **Gunicorn** (Production server)

</div>

---

## 🗂 Project Structure

```bash
Titanic_Project_AkirnogundeDamilare_22CG031827/
│
├── app.py
├── requirements.txt
├── Titanic_hosted_webGUI_link.txt
│
├── model/
│   ├── titanic.csv
│   ├── model_development.py
│   └── titanic_survival_model.pkl
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
