from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# örnek eğitim verileri
texts = [
    "the product was great and i loved it",
    "fast delivery, very satisfied",
    "terrible experience, would not recommend",
    "awful quality, waste of money",
    "amazing service, thank you",
    "did not like it at all, very bad"
]
labels = [1, 1, 0, 0, 1, 0]


# metinleri sayısal vektöre çevir
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# basit bir lojistik regresyon modeli eğit
model = LogisticRegression()
model.fit(X, labels)

import pickle

# modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# vectorizer'ı kaydet
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("Model hazır! Yorum yaz, pozitif mi negatif mi tahmin etsin.")
print("Çıkmak için 'q' yaz.\n")

while True:
   
    comment = input("Enter a review (or type 'q' to quit): ")
    if comment.lower() == "q":
        print("Çıkılıyor...")
        break

    X_new = vectorizer.transform([comment])
    pred = model.predict(X_new)[0]

    if pred == 1:
        print("Prediction: Positive 🙂\n")
    else:
        print("Prediction: Negative 🙁\n")

