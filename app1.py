import streamlit as st
import tweepy
import joblib
import numpy as np
import pandas as pd
import spacy
from fpdf import FPDF

def save_metrics_to_pdf(user_metrics, filename="Twitter_Bot_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Twitter Bot Detection Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for key, value in user_metrics.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    
    pdf.output(filename)
    return filename






# Load the pre-trained model
model = joblib.load('random_forest_bot_detector.pkl')

# Twitter API Authentication
BEARER_TOKEN = "ADD your api key here"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Required feature columns for ML model
columns_needed = [
    'followers_count', 'friends_count', 'favourites_count', 'statuses_count',
    'listed_count', 'BotScore', 'cred', 'normalize_influence',
    'replies', 'retweets',

    # Named Entity Recognition (NER) Features
    "ORG_percentage", "NORP_percentage", "GPE_percentage", "PERSON_percentage",
    "MONEY_percentage", "DATE_percentage", "CARDINAL_percentage",
    "PERCENT_percentage", "ORDINAL_percentage", "FAC_percentage",
    "LAW_percentage", "PRODUCT_percentage", "EVENT_percentage",
    "TIME_percentage", "LOC_percentage", "WORK_OF_ART_percentage",
    "QUANTITY_percentage", "LANGUAGE_percentage",

    # Text-based Features
    "Word count", "Max word length", "Min word length", "Average word length",

    # Part-of-Speech (POS) Features
    "present_verbs", "adjectives", "adverbs", "adpositions", "pronouns",
    "conjunctions", "determiners", "past_verbs", "TOs",

    # Punctuation & Special Character Features
    "dots", "exclamation", "questions", "ampersand",
    "capitals", "digits",

    # Lexical Complexity Features
    "long_word_freq", "short_word_freq"
]

# Function to compute bot score
def compute_bot_score(followers, following, tweet_count, retweets, replies, dot_count, exclamation_count):
    Z = (
        0.3 * np.log1p(followers) +
        0.3 * np.log1p(following) +
        0.2 * (tweet_count / (followers + 1)) +  # High tweet/follower ratio is bot-like
        0.1 * (dot_count + exclamation_count)  # Bots use excessive punctuation
    )
    return round(1 / (1 + np.exp(-Z)), 2)  # Sigmoid normalization

# Function to calculate Credibility Score
def compute_credibility(likes, retweets, replies, tweet_count):
    return round((likes + retweets + replies) / (tweet_count + 1), 2)

# Function to calculate Normalized Influence
def compute_normalized_influence(retweets, replies, followers, tweet_count):
    return round((retweets + replies + followers) / (tweet_count + 1), 2)

# Function to extract text-based, POS, punctuation & NER features
def extract_text_features(texts):
    nlp = spacy.load("en_core_web_sm")

    total_words, max_word_len, min_word_len, avg_word_len = 0, 0, float('inf'), 0
    pos_counts = {
        "present_verbs": 0, "adjectives": 0, "adverbs": 0, "adpositions": 0, "pronouns": 0,
        "conjunctions": 0, "determiners": 0, "past_verbs": 0, "TOs": 0
    }
    punct_counts = {"dots": 0, "exclamation": 0, "questions": 0, "ampersand": 0, "capitals": 0, "digits": 0}
    lexical_complexity = {"long_word_freq": 0, "short_word_freq": 0}
    ner_counts = {k: 0 for k in columns_needed[10:28]}  # Initialize NER features

    for text in texts:
        doc = nlp(text)
        words = [token.text for token in doc if token.is_alpha]

        total_words += len(words)
        if words:
            max_word_len = max(len(w) for w in words)
            min_word_len = min(len(w) for w in words)
            avg_word_len += sum(len(w) for w in words) / len(words)

        # POS features
        for token in doc:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBD", "VBN"]:
                    pos_counts["past_verbs"] += 1
                else:
                    pos_counts["present_verbs"] += 1
            elif token.pos_ == "ADJ":
                pos_counts["adjectives"] += 1
            elif token.pos_ == "ADV":
                pos_counts["adverbs"] += 1

        # Punctuation & special characters
        punct_counts["dots"] += text.count(".")
        punct_counts["exclamation"] += text.count("!")

    return [
        total_words, max_word_len, min_word_len, avg_word_len,
        *pos_counts.values(), *punct_counts.values(), *lexical_complexity.values(), *ner_counts.values()
    ]

# Function to fetch user and tweet metrics
def get_user_and_tweet_metrics(username, tweet_count=10):
    try:
        user = client.get_user(username=username, user_fields=["public_metrics", "description", "created_at"])
        data = user.data

        # Extract user metrics
        followers_count = data.public_metrics["followers_count"]
        friends_count = data.public_metrics["following_count"]
        favourites_count = data.public_metrics.get("like_count", 0)
        statuses_count = data.public_metrics["tweet_count"]
        listed_count = data.public_metrics.get("listed_count", 0)

        user_id = data.id
        tweets = client.get_users_tweets(user_id, max_results=tweet_count, tweet_fields=["created_at", "public_metrics"])

        retweets, replies, likes, dots, exclamations = 0, 0, 0, 0, 0
        text_data = []

        if tweets.data:
            for tweet in tweets.data:
                retweets += tweet.public_metrics["retweet_count"]
                replies += tweet.public_metrics["reply_count"]
                likes += tweet.public_metrics["like_count"]
                text_data.append(tweet.text)

        bot_score = compute_bot_score(followers_count, friends_count, statuses_count, retweets, replies, dots, exclamations)
        credibility = compute_credibility(likes, retweets, replies, statuses_count)
        normalize_influence = compute_normalized_influence(retweets, replies, followers_count, statuses_count)

        text_features = extract_text_features(text_data)

        feature_array = [
            followers_count, friends_count, favourites_count,
            statuses_count, listed_count, bot_score, credibility,
            normalize_influence, replies, retweets
        ] + text_features

        prediction = model.predict([feature_array])
        result = "Bot" if prediction[0] == 1 else "Not a Bot"

        return {
            "Username": username,
            "Followers Count": followers_count,
            "Friends Count": friends_count,
            "Favourites Count": favourites_count,
            "Statuses Count": statuses_count,
            "Listed Count": listed_count,
            "Bot Score": bot_score,
            "Credibility": credibility,
            "Normalized Influence": normalize_influence,
            "Replies": replies,
            "Retweets": retweets,
            "Prediction": result
        }

    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
# Streamlit UI
# Streamlit UI
# === Header Section ===
# === Header Section ===
col1, col2, col3 = st.columns([1, 2, 1])  # Creates three columns for centering

with col2:  # Place logo in the center column
    st.image("final-logo-half.png", use_container_width=False)

st.markdown(
    """
    <div style="text-align: center; padding-top: 0px; margin-top: 0px;height=100px; width=100px;">
    <h1 style="color: #4A90E2; font-size: 32px; margin-top: 0px; padding-top: 0px; "></h1>
    </div>

    """,
    unsafe_allow_html=True
)


st.title("🐦 Twitter Bot Detection App")
username = st.text_input("🔍 Enter Twitter Username:")
tweet_count = st.slider("📊 Number of Recent Tweets to Analyze", min_value=1, max_value=20, value=10)

if st.button("🚀 Analyze"):
    result = get_user_and_tweet_metrics(username, tweet_count)
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
    else:
        st.table(pd.DataFrame(result.items(), columns=["Metric", "Value"]))
        st.success(f"**Prediction:** {'🟢 Not a Bot' if result['Prediction'] == 'Not a Bot' else '🔴 Bot'}")

        # Save the report as a PDF
        pdf_filename = save_metrics_to_pdf(result)
        with open(pdf_filename, "rb") as file:
            st.download_button(
                label="📥 Download Report as PDF",
                data=file,
                file_name=pdf_filename,
                mime="application/pdf"
            )