from flask import Flask, render_template, request
import pickle

# model ve vectorizer yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        user_text = request.form.get("comment")
        X_new = vectorizer.transform([user_text])
        pred = model.predict(X_new)[0]

        if pred == 1:
            prediction = "Positive 🙂"
        else:
            prediction = "Negative 🙁"


    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
