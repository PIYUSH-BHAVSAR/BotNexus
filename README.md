# Twitter Bot Detection App

## Project Overview
The **Twitter Bot Detection App** is a machine learning-powered web application that helps users determine whether a Twitter account is a bot or a real user. This app analyzes various metrics from a given Twitter profile, such as follower count, tweet activity, and linguistic patterns, and predicts the likelihood of the account being a bot.

### Problem Statement
With the increasing presence of automated bots on Twitter, identifying fake or misleading accounts has become essential. Bots are often used for spreading misinformation, manipulating public opinion, and inflating engagement metrics. This app provides users with a simple tool to analyze Twitter profiles and determine their authenticity.

### Key Features
- **Twitter Profile Analysis**: Extracts various user and tweet-related metrics using the Twitter API.
- **Machine Learning Model**: Uses a trained Random Forest classifier to predict whether an account is a bot.
- **Natural Language Processing (NLP)**: Extracts text-based and linguistic features using spaCy.
- **User-Friendly Interface**: Built with Streamlit for an interactive and simple user experience.
- **PDF Report Generation**: Provides a downloadable report summarizing the findings.

---

## Dependencies
To run this project, you need to install the following dependencies:

| Package    | Version |
|------------|---------|
| streamlit  | latest |
| tweepy     | latest |
| joblib     | latest |
| numpy      | latest |
| pandas     | latest |
| spacy      | latest |
| fpdf       | latest |
| Python     | 3.9+   |

### Installation
You can install the required dependencies using:
```bash
pip install streamlit tweepy joblib numpy pandas spacy fpdf
```

Additionally, you need to download the English NLP model for spaCy:
```bash
python -m spacy download en_core_web_sm
```

---

## How to Run the App
1. Clone this repository:
   ```bash
   git clone https://github.com/BotNexus/twitter-bot-detectio.git
   cd twitter-bot-detection
   ```
2. Ensure all dependencies are installed (see the installation section).
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Enter a Twitter username in the input field and click **Analyze** to get bot detection results.

---

## API Authentication
This app uses the Twitter API for fetching user and tweet data. Ensure you have a **Twitter Developer Account** and set up your **Bearer Token** inside `app.py`:
```python
BEARER_TOKEN = "your_twitter_bearer_token"
client = tweepy.Client(bearer_token=BEARER_TOKEN)
```

---

## Model Information
The bot detection model is a **Random Forest classifier** trained on multiple features, including:
- **Account Metadata**: Followers count, friends count, statuses count, listed count.
- **Engagement Metrics**: Retweets, replies, likes.
- **Text Features**: Word count, punctuation frequency, named entity recognition (NER).
- **Part-of-Speech (POS) Features**: Presence of verbs, adjectives, adverbs, etc.

The model is stored in `random_forest_bot_detector.pkl` and is loaded during runtime:
```python
model = joblib.load('random_forest_bot_detector.pkl')
```

---

## Output & PDF Report
Once a Twitter username is analyzed, the app generates a table with the extracted metrics and a **bot prediction**. Users can download a detailed report in PDF format by clicking the **Download Report as PDF** button.

---

## Future Improvements
- Improve accuracy by training on a larger dataset.
- Enhance feature extraction with deep learning models.
- Implement a Flask API for broader integrations.
- Extend support for multilingual text analysis.

---

## Author
- **Your Name**
- **Email:** your.email@example.com
- **GitHub:** [Your GitHub Profile](https://github.com/your-profile)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

