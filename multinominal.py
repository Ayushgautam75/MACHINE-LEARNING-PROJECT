import pandas as pd
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load your dataset
df = pd.read_csv("E:\spam\spam.csv", encoding='ISO-8859-1')[["v1", "v2"]]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text'])
y = df['label']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))
