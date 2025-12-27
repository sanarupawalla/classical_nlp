import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

# START - Helper functions for data cleaning
# THESE FUNCTIONS HAVE TO BE THE SAME AS DEFINED IN THE NOTEBOOK. ANY CHANGE THERE NEEDS TO BE
# CARRIED OUT HERE AS WELL
def to_lowercase(text):
    return str(text).lower()

def remove_html_and_urls(text):
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', text)

    # Remove URLs
    clean = re.sub(r'https?://\S+|www\.\S+', '', clean)
    return clean

def remove_special_chars(text):
    reviews = ''

    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews

def remove_extra_whitespace(text):
    # Replaces multiple spaces with a single space and strips leading/trailing spaces
    clean = re.sub(r'\\s+', ' ', text).strip()
    return clean
# END HELPER FUNCTIONS

def classify_news(inputtext):
    # Clean the text before passing it for prediction
    inputtext = to_lowercase(inputtext)
    inputtext = remove_html_and_urls(inputtext)
    inputtext = remove_special_chars(inputtext)
    inputtext = remove_extra_whitespace(inputtext)

    prediction = model.predict([inputtext])

    # Show the text category according to the order it was set in the notebook
    # 0 - business
    # 1 - entertainment
    # 2 - politics
    # 3 - sport
    # 4 - tech
    if prediction[0] == 0:
        return 'Business'
    if prediction[0] == 1:
        return 'Entertainment'
    if prediction[0] == 2:
        return 'Politics'
    if prediction[0] == 3:
        return 'Sports'
    if prediction[0] == 4:
        return 'Tech'

if __name__ == "__main__":
    # Load the model from the pickled data
    model = pickle.load(open("best_model/text_classification_model.pkl", "rb"))

    # Set the page title
    st.title("News Classification App")

    # Put the instructional text
    st.markdown("""
            This app uses a model trained on a Kaggle dataset consisting of 1490 news articles which contains news articles including their headlines and categories.\n
            In the text box below, input a few words describing news across any of the five categories for which the model is trained:
            **Business, Entertainment, Politics, Sports and Tech**""")
    st.markdown("""
                For example:
                *A new comedy movie has featured on Netflix, India has once again won the cricket world cup, 
                NVDIA has launched a new processor which is capable of running AI much faster.*\n
                """)

    news_text = st.text_area("Input news text", "")

    # Call our functions to take the clean the input and send it to the model for prediction
    if st.button("Classify"):
        label = classify_news(news_text)
        st.success(f"Category of the news is: **{label}**")
