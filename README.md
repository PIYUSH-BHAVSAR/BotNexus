# 🐦 Twitter Bot Detection App

## 🚀 Overview
The **Twitter Bot Detection App** is a **machine learning-powered** web application that analyzes Twitter user profiles and recent tweets to determine whether an account is a bot or a real user.  

It uses **Tweepy** to fetch user data, **Natural Language Processing (NLP)** for text feature extraction, and a **Random Forest model** for classification. Users can enter a Twitter username, analyze their profile, and download a PDF report.

---

## 📌 Features
✅ Fetches **Twitter user metadata** (followers, friends, tweets, likes, etc.)  
✅ Extracts **linguistic features** using **spaCy**  
✅ Computes **bot score, credibility, and influence**  
✅ Uses a **pre-trained Random Forest model** for bot detection  
✅ Generates a **detailed PDF report** with the results  
✅ **Streamlit-powered UI** for easy interaction  

---

## 🛠️ Technologies Used
- **Python**
- **Streamlit**
- **Tweepy** (Twitter API)
- **spaCy** (NLP processing)
- **scikit-learn** (Machine Learning)
- **joblib** (Model persistence)
- **FPDF** (PDF report generation)

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/twitter-bot-detector.git
cd twitter-bot-detector
