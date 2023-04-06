import pytest
import app

text = 'All good'
sent = 'very positive'

def test_analyze_sentiment(text=text):
    output = app.analyze_sentiment(text)
    assert output['sentiment'] == 'very positive'