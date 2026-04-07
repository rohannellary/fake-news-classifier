\# Fake News Detection — BERT × XGBoost Hybrid Classifier



A two-stage NLP pipeline combining BERT embeddings with XGBoost 

for fake news classification, trained on 100,000+ articles.



\## Results

\- 92% accuracy on held-out test set

\- Outperformed single-model baselines by 6pp on macro-F1

\- Fine-tuning reduced false-negative rate by 14% vs frozen-embedding baseline



\## Tech Stack

Python · HuggingFace Transformers (BERT) · XGBoost · scikit-learn



\## Project Structure

\- api\_code.py            → FastAPI endpoint for real-time predictions

\- bert\_preprocessing.py  → Text cleaning and BERT tokenisation

\- bert\_training.py       → BERT fine-tuning and embedding extraction

\- xgboost\_preprocessing.py → Feature preparation for XGBoost

\- xgboost\_training.py    → XGBoost classifier training and evaluation



\## How to Run

pip install -r requirements.txt



\# Step 1 - Preprocess data for BERT

python bert\_preprocessing.py



\# Step 2 - Train BERT and extract embeddings

python bert\_training.py



\# Step 3 - Preprocess for XGBoost

python xgboost\_preprocessing.py



\# Step 4 - Train XGBoost classifier

python xgboost\_training.py



\# Step 5 - Run the API

python api\_code.py



\## Dataset

Trained on 30,000+ training samples and 70,000+ test articles 

across diverse news sources.

