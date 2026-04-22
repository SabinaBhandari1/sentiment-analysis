import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
from flask import Flask, request, render_template

app = Flask(__name__)
data=pd.read_csv("Twitter_Data.csv")
data["clean_text"] = data["clean_text"].fillna("")
data = data.dropna(subset=["category"])
def clean_text(data):
    data=data.lower() #convert whole text into lowercase
    data=re.sub(r'^\w\s','',data) #remove punctuation
    return data

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"], data["category"], test_size=0.2, random_state=42
)

# Fit vectorizer on TRAINING data only
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)  # only transform, not fit

# Balance classes - give more weight to underrepresented classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train model
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))
print("Class distribution:", data["category"].value_counts())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    text = clean_text(text)
    vect = vectorizer.transform([text])
    result = model.predict(vect)[0]

    sentiment_map = {
        1: "Positive",
        0: "Neutral",
        -1: "Negative"
    }
    sentiment = sentiment_map[result]  # converts -1/0/1 to label

    return render_template("index.html", prediction=sentiment)
if __name__ == "__main__":
    app.run(debug=True)




    

