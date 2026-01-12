import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['label', 'message']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


svm_linear = SVC(kernel='linear', class_weight='balanced')
svm_linear.fit(X_train_tfidf, y_train)


accuracy = accuracy_score(y_test, svm_linear.predict(X_test_tfidf))
print("Model accuracy:", accuracy)


joblib.dump(svm_linear, "svm_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved.")
