# Sentiment Analysis API

An API for sentiment analysis using FastAPI and [BERT-base-multilingual-uncased-sentiment model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment).


Deploy API -->
```
uvicorn app:app --reload
```

Test using browser example -->
```
http://127.0.0.1:8000/sentiment_analysis/?text="wow%20great"
```
response -->
```
{"sentiment":"very positive","probability":0.7995468378067017}
```