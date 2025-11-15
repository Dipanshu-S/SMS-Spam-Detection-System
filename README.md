# SMS Spam Detection System

A machine learning-powered SMS spam classifier that uses Natural Language Processing (NLP) to detect and filter spam messages with high accuracy.

## What is this project?

This SMS Spam Detection System is an intelligent text classification application that analyzes SMS messages and determines whether they are spam (unsolicited/promotional messages) or ham (legitimate messages). The system employs Naive Bayes classification algorithms and TF-IDF vectorization to achieve over 96% accuracy in spam detection.

### Key Features

- **High Accuracy**: Achieves 96.4% accuracy with 94.5% precision using Multinomial Naive Bayes
- **Real-time Detection**: Instant classification of SMS messages through a user-friendly Streamlit interface
- **Comprehensive Text Processing**: Advanced NLP pipeline including tokenization, stopword removal, and stemming
- **Interactive Web Interface**: Clean, intuitive UI for easy message testing

## Why this project exists

SMS spam is a persistent problem that wastes time, poses security risks, and can lead to financial fraud. This project provides:

1. **Protection**: Helps users identify potentially malicious or unwanted messages
2. **Learning Resource**: Demonstrates practical applications of NLP and machine learning
3. **Efficiency**: Automates the tedious process of manually filtering spam
4. **Scalability**: Can be integrated into larger communication systems

## How to use this project

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
git clone https://github.com/Dipanshu-S/SMS-Spam-Detection-System.git
cd SMS-Spam-Detection-System

text

2. **Install required dependencies**
pip install -r requirements.txt

text

### Running the Application

**Launch the Streamlit web interface**
streamlit run Spam_Detector.py

text

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. Enter an SMS message in the text input field
2. Click the **"Predict"** button
3. The system will display whether the message is "Spam" or "Not Spam"

### Example Messages

**Spam Example**:
WINNER!! You have been selected to receive a £1000 prize.
Call 09061701461 to claim. Valid 12 hours only.

text

**Ham Example**:
Hey, are you free for dinner tonight? Let me know!

text

## Project Structure

SMS-Spam-Detection-System/
├── SMS-Spam Detection.ipynb # Model training and data analysis notebook
├── Spam_Detector.py # Streamlit web application
├── model.pkl # Trained Multinomial Naive Bayes model
├── vectorizer.pkl # TF-IDF vectorizer for text transformation
├── requirements.txt # Python dependencies
├── sms-spam(in).csv # Dataset (5,572 messages)
└── README.md # This file

text

## Model Details

### Dataset
- **Total Messages**: 5,572 SMS messages
- **Ham Messages**: 4,825 (86.6%)
- **Spam Messages**: 747 (13.4%)
- **Source**: SMS Spam Collection Dataset

### Text Processing Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Split messages into individual words
3. **Cleaning**: Remove special characters and numbers
4. **Stopword Removal**: Filter out common English stopwords
5. **Stemming**: Reduce words to their root form using Porter Stemmer

### Model Performance

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| Multinomial Naive Bayes | 96.4% | 94.5% |
| Bernoulli Naive Bayes | 96.4% | 94.5% |
| Gaussian Naive Bayes | 97.0% | 87.7% |

*Multinomial Naive Bayes selected for optimal precision-accuracy balance*

## Dependencies

streamlit
scikit-learn
numpy
scipy
nltk==3.8.1

text

Install all dependencies with:
pip install -r requirements.txt

text

## Technical Stack

- **Programming Language**: Python 3.x
- **Machine Learning**: scikit-learn (Multinomial Naive Bayes)
- **NLP**: NLTK (Natural Language Toolkit)
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas, SciPy

## Common Spam Indicators

The model identifies these common spam keywords:
- "CALL", "FREE", "WIN", "PRIZE", "URGENT"
- "CLAIM", "GUARANTEED", "CASH", "REWARD"
- Excessive use of numbers and special characters
- Unusual capitalization patterns

## Future Enhancements

- [ ] Add multi-language support
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Create mobile application
- [ ] Add user feedback loop for continuous improvement
- [ ] Deploy to cloud platform (Heroku, AWS, or Azure)

## Author

**Dipanshu Shamkuwar**  
Engineering Student | AI/ML Enthusiast | WCEM

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- Dataset: UCI Machine Learning Repository - SMS Spam Collection
- NLTK for natural language processing tools
- scikit-learn for machine learning algorithms
- Streamlit for the web interface framework

---

**Note**: This system is designed for educational purposes. For production use, consider additional security measures and regular model retraining with updated data.
