# Sentiment Analysis Web Application

This project is a simple machine learning web application that predicts whether a given text review is **positive** or **negative**.  
The model is trained using a small manually created English dataset and the predictions are displayed through a Flask-based web interface.

## 🔍 Features
- Real-time sentiment prediction (positive / negative)
- English-language dataset
- Logistic Regression machine learning model
- Text preprocessing using CountVectorizer
- Web interface built with Flask (HTML + CSS)
- Clean and simple project structure

## 🛠 Technologies Used
- Python
- scikit-learn
- Flask
- HTML & CSS
- pickle (model saving & loading)

## 🚀 How to Run the Project

### 1. Install dependencies
Make sure Python is installed. Then run:

```bash
pip install flask scikit-learn
```

### 2. Start the web server

```bash
python app.py
```

### 3. Open the app in your browser

```
http://127.0.0.1:5000/
```

You can now write a review and instantly see whether the model predicts **Positive** or **Negative**.

---

## 📂 Project Structure

```
sentiment-project/
│
├── app.py
├── sentiment_app.py
├── model.pkl
├── vectorizer.pkl
├── README.md
│
└── templates/
       └── index.html
```

- **sentiment_app.py** → Script used to train the model  
- **app.py** → Flask backend (web server)  
- **templates/index.html** → Web interface  
- **model.pkl / vectorizer.pkl** → Saved ML model and preprocessing tools  

---

## 🌱 Future Ideas / Improvements
- Train with a larger real-world English dataset  
- Add accuracy metrics & evaluation results  
- Improve UI/UX design  
- Convert into a REST API returning JSON  
- Deploy the app online (Render / Railway / Vercel)  

---

## 📌 About This Project
This is an entry-level Artificial Intelligence project built to practice:
- Machine Learning fundamentals  
- Text preprocessing  
- Model training and saving  
- Flask web development  
- Project structuring  
- Documentation and GitHub publishing  

It represents my first full ML workflow from training to deployment.
