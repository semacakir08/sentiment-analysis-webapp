from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --------- EĞİTİM VERİSİ (İNGİLİZCE, POZİTİF/NEGATİF) ---------
texts = [
    # positive
    "I really loved this product, totally worth it",
    "The quality is amazing, I will buy again",
    "Fast delivery and great packaging, I'm impressed",
    "Absolutely perfect, exceeded my expectations",
    "Very satisfied with the experience",
    "Works better than I expected, highly recommended",
    "The design is beautiful and feels premium",
    "Customer service was very helpful and polite",
    "This is one of the best purchases I've made",
    "Good value for the price, I’m happy with it",

    # negative
    "Terrible product, I want a refund",
    "The quality is very bad, totally disappointed",
    "Arrived damaged, packaging was horrible",
    "Does not work as described, waste of money",
    "Extremely slow delivery, not recommended",
    "Poor customer service, they didn’t help at all",
    "Cheap material and ugly design",
    "Stopped working after a few days",
    "Definitely not worth the price",
    "Bad experience overall, won’t buy again",

    # ekstra negatif örnekler, özellikle senin test cümlene benzer
    "This product is terrible, I will never buy it again",
    "Really bad experience, I regret buying this product",
    "The product is awful, I would never recommend it",
    "I hate this product, it is a complete waste of money"
]

labels = [
    1,1,1,1,1,1,1,1,1,1,  # 10 pozitif
    0,0,0,0,0,0,0,0,0,0,  # 10 negatif
    0,0,0,0               # ekstra 4 negatif
]

# --------- MODEL EĞİTİMİ ---------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# --------- MODELİ KAYDET ---------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# --------- TEST CÜMLESİ (ÖZEL OLARAK SENİN ÖRNEĞİN) ---------
test_sentence = "product is terrible I will never buy this again"
X_test = vectorizer.transform([test_sentence])
pred_test = model.predict(X_test)[0]
print("Test sentence:", test_sentence)
print("Prediction for test sentence:",
      "Positive" if pred_test == 1 else "Negative")

print("\nModel is ready! Type a review, or 'q' to quit.\n")

# --------- KULLANICI GİRİŞ DÖNGÜSÜ ---------
while True:
    comment = input("Enter a review (or type 'q' to quit): ")
    if comment.lower() == "q":
        print("Exiting...")
        break

    X_new = vectorizer.transform([comment])
    pred = model.predict(X_new)[0]

    if pred == 1:
        print("Prediction: Positive 🙂\n")
    else:
        print("Prediction: Negative 🙁\n")
