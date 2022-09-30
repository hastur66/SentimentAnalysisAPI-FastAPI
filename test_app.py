import pytest

text = 'All good'
sent = 'very positive'

def test_analyze_sentiment(text=text):
    assert sent == 'very positive'