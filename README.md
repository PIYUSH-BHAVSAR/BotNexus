# Twitter Bot Detection App

## Project Overview
The **Twitter Bot Detection App** is a machine learning-powered application designed to analyze Twitter accounts and determine the likelihood of them being bots. The app fetches user and tweet data, extracts various features, and classifies the account as either a **bot** or **not a bot**.

### Problem Statement
Social media platforms, including Twitter, are often flooded with automated bot accounts that spread misinformation, manipulate public opinion, or engage in spam activities. This project aims to provide a reliable tool to detect such bot accounts and enhance the credibility of online interactions.

### Key Features
- Fetches real-time Twitter user data using the **Tweepy API**.
- Analyzes key metrics such as followers, tweets, likes, and engagement.
- Utilizes **Natural Language Processing (NLP)** to analyze text features.
- Implements a **Random Forest** machine learning model for classification.
- Generates detailed reports and allows users to download results as a **PDF**.
- Interactive UI built with **Streamlit**.
- Provides a demo video for better understanding.

### Demo Video
🔗 [Watch the Demo](https://www.loom.com/share/d30b61759a484fef96954f8d8b7fd496)

---

## Dependencies
To run this project, you need the following libraries and packages:

```bash
pip install streamlit tweepy joblib numpy pandas spacy fpdf
```

| Dependency | Version |
|------------|---------|
| Python     | 3.9+    |
| Streamlit  | latest |
| Tweepy     | latest |
| Joblib     | latest |
| NumPy      | latest |
| Pandas     | latest |
| spaCy      | latest |
| fpdf       | latest |

Ensure you have **spaCy's English model** installed using:
```bash
python -m spacy download en_core_web_sm
```

---

## Installation & Usage
1. Clone the repository:
   ```bash
   https://github.com/PIYUSH-BHAVSAR/BotNexus.git
   cd twitter-bot-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Enter a Twitter username in the UI and analyze the account.

---

## Model & Prediction
The model is a **Random Forest classifier** trained on a dataset containing bot and human Twitter accounts. It considers:
- User account metrics (followers, tweets, likes, retweets, etc.).
- NLP-based text features (word length, entity recognition, etc.).
- Engagement metrics (replies, retweets, credibility score, etc.).

After processing, the app predicts if the account is a **bot** or **not a bot**.

---

## Report Generation
The app generates a **detailed report** with:
- User engagement statistics.
- NLP-based text feature analysis.
- Bot detection prediction.
- **Downloadable PDF report** for further analysis.

---

## Contributing
Feel free to contribute to the project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact
For any queries or collaborations, reach out to:
📧 Email: piyushbhavsar1665@gmail.com
🐦 Linkedin: https://www.linkedin.com/in/piyush-bhavsar

