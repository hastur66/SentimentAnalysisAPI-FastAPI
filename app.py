from transformers import pipeline
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

nlp = pipeline(task='sentiment-analysis', 
               model='nlptown/bert-base-multilingual-uncased-sentiment')

@app.get('/')
def get_root():
    return {'message': 'This is a sentiment analysis API'}
    

@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)

def analyze_sentiment(text):
    """model perdiction"""
    
    result = nlp(text)

    if result[0]['label'] == '1 star':
        sent = 'very negative'
    elif result[0]['label'] == '2 star':
        sent = 'negative'
    elif result[0]['label'] == '3 star':
        sent = 'neutral'
    elif result[0]['label'] == '4 start':
        sent = 'positive'
    else:
        sent = 'very positive'
    prob = result[0]['score']

    return {'sentiment': sent, 'probability': prob}
    
if __name__=='__main__':
    uvicorn.run(app)