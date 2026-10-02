import streamlit as st
import nltk
import spacy
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK + Spacy Data
nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="TF-IDF & Embeddings", layout="wide")
st.title("TF-IDF & Word Embedding using Regex")

text = st.text_area("Enter your text", height=150,
                    placeholder="Example: NLP is amazing! It helps machines understand language.")

option = st.sidebar.radio("Select Technique", [
    "Regex Cleaning",
    "TF-IDF",
    "Word Embedding (Spacy)"
])

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt)
    txt = re.sub(r'\S+@\S+\.\S+', '', txt)
    txt = re.sub(r'<.*?>', '', txt)
    txt = re.sub(r'[\U00010000-\U0010ffff]', '', txt)
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r'[^a-z\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

if st.button("Process"):
    if text.strip() == "":
        st.warning("Please enter text!")
    
    elif option == "Regex Cleaning":
        cleaned = clean_text(text)
        doc = nlp(cleaned)
        words = [t.text for t in doc if not t.is_stop]

        st.subheader("Original Text")
        st.write(text)

        st.subheader("Cleaned Text")
        st.write(" ".join(words))
    
    elif option == "TF-IDF":
        cleaned = clean_text(text)
        corpus = [cleaned]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names_out()
        values = X.toarray()[0]

        df = pd.DataFrame({"Word": features, "TF-IDF": values})
        df = df.sort_values(by="TF-IDF", ascending=False)
        st.dataframe(df, use_container_width=True)

    elif option == "Word Embedding (Spacy)":
        cleaned = clean_text(text)
        doc = nlp(cleaned)
        words = [t.text for t in doc if not t.is_stop]

        if len(words) == 0:
            st.warning("No valid words for embedding after cleaning!")
        else:
            vectors = [nlp(w).vector for w in words]
            df = pd.DataFrame(vectors, index=words)
            st.write("### Word Embedding Vectors")
            st.dataframe(df, use_container_width=True)

