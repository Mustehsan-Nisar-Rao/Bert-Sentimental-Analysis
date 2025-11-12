# BERT Customer Feedback Sentiment Analysis

A **BERT-based sentiment classification** project that predicts whether customer feedback is **Positive** or **Negative**.

## 📂 Dataset
- Source: [Kaggle Customer Feedback Dataset](https://www.kaggle.com/datasets/vishweshsalodkar/customer-feedback-dataset?select=sentiment-analysis.csv)
- Contains textual feedback and sentiment labels.

## 🛠️ Project Components
1. **Preprocessing & Tokenization**
   - Cleaned text data and tokenized using `BERT tokenizer`.
   - Converted text to token IDs and attention masks for model input.

2. **Training & Validation**
   - Fine-tuned `bert-base-uncased` for sentiment classification.
   - Split dataset into training and validation sets.
   - Implemented evaluation metrics: accuracy, F1-score, and confusion matrix.

3. **Evaluation**
Accuracy: 0.90
F1-Score: 0.90
Confusion Matrix:
[[9 0]
[2 9]]

4. **Prediction**
   - Input any customer feedback and get sentiment predictions in real-time.

## 🚀 Live Demo
Try the model live using **Streamlit**:  
[https://bert-sentimental-analysis-8axuvw62jleqytbeuz2bn9.streamlit.app/](https://bert-sentimental-analysis-8axuvw62jleqytbeuz2bn9.streamlit.app/)

## 💻 Code Repository
[GitHub Repository](https://github.com/Mustehsan-Nisar-Rao/Bert-Sentimental-Analysis)

## 📈 Results
- Accuracy: **90%**  
- F1-Score: **0.90**  
- Confusion Matrix:
[[9 0]
[2 9]]

## 📝 Example Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("path_to_finetuned_model")

text = "The product quality was excellent!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_label = torch.argmax(outputs.logits, dim=1)
