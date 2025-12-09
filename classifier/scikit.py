import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



path ="data/CulturalDeepfake/posts"

# carica i commenti (colonne: "text", "label")
df = pd.read_csv(f"{path}/1.csv")

# suddivisione train/test
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# vettorizzazione
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# classificatore
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# valutazione
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))
