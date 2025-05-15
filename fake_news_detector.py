import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/news.csv")  # The dataset should have columns like ['text', 'label']

# Data preprocessing
df = df.dropna()
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})  # adjust as per your dataset

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Text vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Function to predict new input
def predict_news(news_text):
    vec = tfidf.transform([news_text])
    pred = model.predict(vec)
    return "Fake" if pred[0] else "Real"

# Test example
print(predict_news("President announces new stimulus plan for economy"))
