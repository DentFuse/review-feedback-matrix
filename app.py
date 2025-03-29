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
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify

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

    def _preprocess_text(self, text):
        """Preprocess text by lowercasing, removing punctuation, and removing stopwords."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return tokens

    def _calculate_word_overlap(self, description, review):
        """Calculate the percentage of words from description that appear in the review."""
        desc_tokens = set(self._preprocess_text(description))
        review_tokens = set(self._preprocess_text(review))
        overlap = len(desc_tokens.intersection(review_tokens))
        total_desc_words = len(desc_tokens)
        overlap_percentage = (overlap / total_desc_words) * 100 if total_desc_words > 0 else 0
        return overlap_percentage

    def _sentiment_analysis(self, text):
        """Perform sentiment analysis using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.3:
            return 'very_positive', polarity
        elif 0.1 < polarity <= 0.3:
            return 'positive', polarity
        elif -0.1 <= polarity <= 0.1:
            return 'neutral', polarity
        elif -0.3 <= polarity < -0.1:
            return 'negative', polarity
        else:
            return 'very_negative', polarity

    def _classify_review(self, review, description):
        """Classify review based on word overlap and sentiment."""
        overlap_percentage = self._calculate_word_overlap(description, review)
        sentiment_type, sentiment_score = self._sentiment_analysis(review)
        
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
            _, sentiment_score = self._sentiment_analysis(review)
            overlap = self._calculate_word_overlap(description, review)
            
            review_data = {
                'text': review,
                'sentiment': sentiment_score,
                'category': category,
                'overlap': overlap
            }
            
            self.matrix[category].append(review_data)
            classified_reviews.append(review_data)
        
        return classified_reviews

# Flask Application
app = Flask(__name__)
feedback_matrix = FeedbackMatrix()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form
        description = request.form['description']
        reviews = [review.strip() for review in request.form['reviews'].split('\n') if review.strip()]
        
        # Analyze reviews
        classified_reviews = feedback_matrix.analyze_reviews(description, reviews)
        
        # Prepare data for visualizations
        df = pd.DataFrame(classified_reviews)
        
        # Pie Chart of Category Distribution
        category_counts = df['category'].value_counts()
        pie_chart = go.Figure(data=[go.Pie(
            labels=category_counts.index.tolist(), 
            values=category_counts.values.tolist(), 
            hole=.3
        )])
        pie_chart_json = json.dumps(pie_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Define color mapping
        color_map = {
            'Positive Expected': 'green',
            'Positive Unexpected': 'lightgreen',
            'Negative Expected': 'red',
            'Negative Unexpected': 'lightcoral'
        }

        # Ensure categories are properly mapped to colors
        df['color'] = df['category'].map(color_map).fillna('gray')

        # Ensure x and y values are properly formatted
        df = df.dropna(subset=['overlap', 'sentiment'])  # Drop rows with missing values
        x_values = df['overlap'].astype(float).tolist()  # Convert to float and list
        y_values = df['sentiment'].astype(float).tolist()  # Convert to float and list

        # Add random variance to x-axis values
        x_values = [x + np.random.uniform(-0.35, 0.35) for x in x_values]  # Adds random noise in range [-0.25, 0.25]

        scatter_plot = go.Figure(data=go.Scatter(
            x=x_values, 
            y=y_values, 
            mode='markers',
            text=df['category'].tolist(),  # Ensure text is a list
            marker=dict(
                size=10,
                color=df['color'].tolist(),  # Convert colors to list
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
            scatter_plot=scatter_plot_json
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)