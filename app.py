import os
import re
import nltk
import json
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
import ssl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from amazon import getAmazonDescAndReviews

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class FeedbackMatrix:
    def __init__(self):
        self.categories = [
            'Positive Expected', 
            'Positive Unexpected', 
            'Negative Expected', 
            'Negative Unexpected'
        ]
        self.matrix = {cat: [] for cat in self.categories}
        self.product_description = ""
        self.stop_words = set(stopwords.words('english'))
        self.amazon_lexicon = None

    def _preprocess_text(self, text):
        """Preprocess text by lowercasing, removing punctuation, and removing stopwords."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def load_amazon_lexicon(self, file_path):
        """
        Loads Amazon review sentiment data from a ground truth file.
        
        Args:
            file_path (str): Path to the amazonReviewSnippets_GroundTruth.txt file
            
        Returns:
            dict: A dictionary mapping review text to their ground truth sentiment scores
        """
        if self.amazon_lexicon:
            return self.amazon_lexicon
        
        sentiment_dict = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Assuming the file is tab-separated with review text and sentiment score
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        review_text = parts[0]
                        sentiment_score = float(parts[1])
                        sentiment_dict[review_text] = sentiment_score
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"Error reading file: {e}")
        print(f"Loaded {len(sentiment_dict)} reviews from Amazon ground truth file")
        self.amazon_lexicon = sentiment_dict
        return self.amazon_lexicon

    def _analyze_product_review(self, review_text):
        """
        Analyzes the sentiment of a product review and returns a score from -1 to 1.
        
        Args:
            review_text (str): A 1-2 sentence product review
            
        Returns:
            float: A sentiment score between -1 (extremely negative) and 1 (extremely positive)
        """
        amazon_file_path = "amazonReviewSnippets_GroundTruth.txt"
        amazon_lexicon = self.load_amazon_lexicon(amazon_file_path)
        if amazon_lexicon and review_text in amazon_lexicon:
            return amazon_lexicon[review_text]
        
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(review_text)
        return sentiment_dict['compound']

    def _calculate_word_overlap(self, description, review):
        """Calculate the percentage of words from description that appear in the review."""
        desc_tokens = set(self._preprocess_text(description))
        review_tokens = set(self._preprocess_text(review))
        overlap = len(desc_tokens.intersection(review_tokens))
        total_desc_words = len(desc_tokens)
        overlap_percentage = (overlap / total_desc_words) * 100 if total_desc_words > 0 else 0
        return overlap_percentage

    def _get_sentiment(self, polarity):
        if polarity > 0.3:
            return 'very_positive', polarity
        elif 0.1 < polarity <= 0.3:
            return 'positive', polarity
        elif -0.1 <= polarity <= 0.1:
            return 'negative', polarity
        elif -0.3 <= polarity < -0.1:
            return 'negative', polarity
        else:
            return 'very_negative', polarity

    def _classify_review(self, review, description):
        """Classify review based on word overlap and sentiment."""
        overlap_percentage = self._calculate_word_overlap(description, review)
        sentiment_type, sentiment_score = self._get_sentiment(self._analyze_product_review(review))
        
        if sentiment_type in ['very_positive', 'positive']:
            return 'Positive Expected' if overlap_percentage > 0 else 'Positive Unexpected'
        elif sentiment_type in ['very_negative', 'negative']:
            return 'Negative Expected' if overlap_percentage > 0 else 'Negative Unexpected'
        else:
            return 'Positive Expected'

    def analyze_reviews(self, description, reviews):
        """Analyze reviews and return detailed classification."""
        self.product_description = description
        self.matrix = {cat: [] for cat in self.categories}
        
        classified_reviews = []
        for review in reviews:
            category = self._classify_review(review, description)
            _, sentiment_score = self._get_sentiment(self._analyze_product_review(replace_non_alphanumeric(review)))
            overlap = self._calculate_word_overlap(description, replace_non_alphanumeric(review))
            if sentiment_score == 0: 
                continue
            review_data = {
                'text': review,
                'sentiment': sentiment_score,
                'category': category,
                'overlap': overlap
            }
            self.matrix[category].append(review_data)
            classified_reviews.append(review_data)
        
        return classified_reviews

def replace_non_alphanumeric(text):
    """
    Replaces all non-alphanumeric characters in a given text with spaces.
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: The cleaned text with non-alphanumeric characters replaced by spaces
    """
    if not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return cleaned_text

# Flask Application
app = Flask(__name__)
feedback_matrix = FeedbackMatrix()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        amazon_url = request.form.get('amazon_url', '').strip()
        if amazon_url:
            # If Amazon URL provided, fetch data from it
            title, description, reviews = getAmazonDescAndReviews(amazon_url)
        else:
            # Otherwise, get description and reviews from form inputs
            description = replace_non_alphanumeric(request.form['description'])
            reviews = [review.strip() for review in request.form['reviews'].split('\n') if review.strip()]

        # Run analysis
        classified_reviews = feedback_matrix.analyze_reviews(description, reviews)
        
        # Prepare dataframe and visualizations
        df = pd.DataFrame(classified_reviews)
        
        category_counts = df['category'].value_counts()
        pie_chart = go.Figure(data=[go.Pie(
            labels=category_counts.index.tolist(), 
            values=category_counts.values.tolist(), 
            hole=.3
        )])
        pie_chart_json = json.dumps(pie_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        color_map = {
            'Positive Expected': 'green',
            'Positive Unexpected': 'lightgreen',
            'Negative Expected': 'red',
            'Negative Unexpected': 'lightcoral'
        }
        df['color'] = df['category'].map(color_map).fillna('gray')
        df = df.dropna(subset=['overlap', 'sentiment'])
        x_values = df['overlap'].astype(float).tolist()
        y_values = df['sentiment'].astype(float).tolist()
        x_values = [x + np.random.uniform(-0.35, 0.35) for x in x_values]

        scatter_plot = go.Figure(data=go.Scatter(
            x=x_values, 
            y=y_values, 
            mode='markers',
            text=df['category'].tolist(),
            marker=dict(
                size=10,
                color=df['color'].tolist(),
                opacity=0.7
            )
        ))
        scatter_plot.update_layout(
            title='Sentiment vs Word Overlap',
            xaxis_title='Word Overlap (%)',
            yaxis_title='Sentiment Score'
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'index.html',
            reviews=classified_reviews,
            pie_chart=pie_chart_json,
            scatter_plot=scatter_plot_json,
            description=description,
            reviews_text='\n'.join(reviews),
            amazon_url=amazon_url,
            lock_description=bool(amazon_url),  # True if amazon_url given
            lock_amazon=bool(description and reviews and not amazon_url),  # True if desc/reviews given
        )
    
    # GET method - empty form initially
    return render_template('index.html', description='', reviews_text='', amazon_url='')

if __name__ == '__main__':
    app.run(debug=True)
