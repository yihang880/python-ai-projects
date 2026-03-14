import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def scrape_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except requests.exceptions.RequestException as e:
        print(f"Error during web scraping: {e}")
        return None

def analyze_sentiment(text):
    if not text:
        return None
    analysis = TextBlob(text)
    # Polarity is a float within the range [-1.0, 1.0] where -1 is negative and 1 is positive.
    # Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    return {
        "polarity": analysis.sentiment.polarity,
        "subjectivity": analysis.sentiment.subjectivity
    }

if __name__ == "__main__":
    # Example usage:
    # For TextBlob, you might need to download its NLTK data first:
    # import nltk
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('brown')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    
    example_url = "https://www.bbc.com/news/world-us-canada-68557339" # Replace with a real article URL
    article_content = scrape_article(example_url)
    
    if article_content:
        sentiment = analyze_sentiment(article_content)
        if sentiment:
            print(f"Article URL: {example_url}")
            print(f"Sentiment Polarity: {sentiment["polarity"]:.2f} (Negative: -1.0, Positive: 1.0)")
            print(f"Sentiment Subjectivity: {sentiment["subjectivity"]:.2f} (Objective: 0.0, Subjective: 1.0)")
        else:
            print("Could not analyze sentiment.")
    else:
        print("Could not scrape article content.")
