# 📩 SMS Spam Detection System  
*A smart ML-powered spam classifier with an interactive web UI*

<div align="center">

🔍 **Machine Learning + NLP + Streamlit**  
⚡ Accurate • 🧠 Lightweight • 🌐 Real-time

</div>

---

## 🚀 Overview
This project is an intelligent SMS spam classifier that detects whether a message is **Spam** or **Ham** using **Naive Bayes** and **TF-IDF** vectorization.  
It achieves **96%+ accuracy** and includes a clean, interactive **Streamlit UI**.

---

## ✨ Features
- 🎯 **High Accuracy** — 96.4% accuracy, 94.5% precision  
- ⚡ **Real-Time Detection** — Instant prediction through a web UI  
- 🧹 **NLP Pipeline** — Tokenization, stopword removal, stemming  
- 💻 **Interactive Interface** — Simple and beginner-friendly UI  
- 📦 **Production Ready** — Easy to integrate into bigger systems  

---

## 🧠 Why This Project?
SMS spam leads to: ✔ Time waste ✔ Phishing ✔ Security risks ✔ Fraud  
This system helps with:  

1. 🛡 User protection  
2. 📘 Learning practical NLP  
3. ⚙ Automated spam filtering  
4. 📈 Scalable integration  

---

## 📦 Installation Guide

### 🔧 Prerequisites
- Python 3.8+  
- pip  

### 📥 Clone the Repository
```bash
git clone https://github.com/Dipanshu-S/SMS-Spam-Detection-System.git
cd SMS-Spam-Detection-System
````

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run Spam_Detector.py
```

The app will open at: **[http://localhost:8501](http://localhost:8501)**

---

## 🖥 How to Use

1. Type an SMS message
2. Click **Predict**
3. View result → **Spam** 🚫 or **Not Spam** ✅

---

## ✉️ Example Messages

### 🚫 Spam

> WINNER!! You have been selected for a £1000 prize. Call now to claim.

### ✅ Ham

> Hey, are you free for dinner tonight? Let me know!

---

## 📁 Project Structure

```
SMS-Spam-Detection-System/
│── SMS-Spam Detection.ipynb     # Training & analysis notebook
│── Spam_Detector.py             # Streamlit application
│── model.pkl                    # Trained Naive Bayes model
│── vectorizer.pkl               # TF-IDF vectorizer
│── requirements.txt             # Dependencies
│── sms-spam(in).csv             # Dataset (5,572 messages)
└── README.md                    # Documentation
```

---

## 📊 Model Details

### 📁 Dataset Summary

* **Total:** 5,572 messages
* **Ham:** 4,825
* **Spam:** 747
* **Source:** UCI SMS Spam Dataset

### 🔤 NLP Pipeline

1. Lowercasing
2. Tokenization
3. Remove special chars
4. Stopword removal
5. Stemming

### 🧪 Model Performance

| Algorithm          | Accuracy  | Precision |
| ------------------ | --------- | --------- |
| **Multinomial NB** | **96.4%** | **94.5%** |
| Bernoulli NB       | 96.4%     | 94.5%     |
| Gaussian NB        | 97.0%     | 87.7%     |

📝 *Multinomial NB chosen for best balanced results.*

---

## 🛠 Tech Stack

* 🐍 Python
* 🤖 Scikit-learn
* 🧠 NLTK
* 🌐 Streamlit
* 📊 NumPy, Pandas, SciPy

---

## 🚨 Common Spam Indicators

* "FREE", "WIN", "PRIZE", "CALL NOW", "CLAIM"
* Excessive digits or symbols
* Strange capitalization
* Urgent limited-time offers

---

## 🔮 Future Enhancements

* [ ] Multi-language support
* [ ] LSTM/BERT models
* [ ] Mobile app
* [ ] User feedback loop
* [ ] Cloud deployment

---

## 👤 Author

**Dipanshu Shamkuwar**
AI/ML Enthusiast • Engineering Student • WCEM

---

## 📜 License

Open-source — free for learning & research.

---

## 🙏 Acknowledgments

* UCI SMS Spam Collection
* NLTK team
* Scikit-learn
* Streamlit

---
