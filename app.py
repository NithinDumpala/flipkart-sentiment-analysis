import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Text cleaning setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("Flipkart Product Review Sentiment Analysis")

st.write("Enter a product review below to analyze sentiment.")

user_input = st.text_area("Review Text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(user_input)
        vector = tfidf.transform([cleaned_review])
        prediction = model.predict(vector)[0]

        if prediction == "Positive":
            st.success("Sentiment: Positive 😊")
        else:
            st.error("Sentiment: Negative 😞")
