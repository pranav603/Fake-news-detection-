import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from Google Drive CSV links
@st.cache_data
def load_data():
    true_url = "https://drive.google.com/uc?id=1HRN_529qHIopnCSK3mfAJxoAOWIKbgtz"
    fake_url = "https://drive.google.com/uc?id=1eWNN22mAeCOv8xKnsLo-HsEfXVoREkZc"
    
    true_news = pd.read_csv(true_url)
    fake_news = pd.read_csv(fake_url)

    true_news['label'] = 0
    fake_news['label'] = 1

    df = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)
    df = df[['text', 'label']]
    return df

df = load_data()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app UI
st.title("Fake News Detector")

st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

user_input = st.text_area("Enter news text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        result = "Fake News 🚫" if prediction == 1 else "Real News ✅"
        st.success(f"Prediction: {result}")
