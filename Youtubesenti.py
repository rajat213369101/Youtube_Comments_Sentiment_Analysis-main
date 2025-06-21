import csv
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO


def extract_video_id(youtube_link):
    video_id_regex = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_regex, youtube_link)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None
    
# Function to generate suggested comments for the creator
def generate_suggested_comments(comments):
    sid = SentimentIntensityAnalyzer()
    suggested_comments = []

    for comment in comments:
        sentiment_scores = sid.polarity_scores(comment)
        # Suggest comments with high negative sentiment
        if sentiment_scores['compound'] < -0.5:
            suggested_comments.append(comment)

    return suggested_comments

# Function to perform sentiment analysis
# Modify the analyze_sentiment function to return comments with sentiment scores and counts
def analyze_sentiment(csv_file):
    sid = SentimentIntensityAnalyzer()
    comments = []

    with open(csv_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row['Comment'])

    num_neutral = 0
    num_positive = 0
    num_negative = 0

    for comment in comments:
        sentiment_scores = sid.polarity_scores(comment)
        if sentiment_scores['compound'] == 0.0:
            num_neutral += 1
        elif sentiment_scores['compound'] > 0.0:
            num_positive += 1
        else:
            num_negative += 1

    results = {'num_neutral': num_neutral, 'num_positive': num_positive, 'num_negative': num_negative}
    return comments,results

# Function to plot results
def generate_bar_chart(results):
    num_neutral = results.get('num_neutral', 0)
    num_positive = results.get('num_positive', 0)
    num_negative = results.get('num_negative', 0)

    df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Number of Comments': [num_positive, num_negative, num_neutral]
    })

    fig = px.bar(
        df,
        x='Sentiment',
        y='Number of Comments',
        color='Sentiment',
        color_discrete_sequence=['#87CEFA', '#FFA07A', '#D3D3D3'],  # Updated color scheme
        labels={'Number of Comments': 'Number of Comments'},
    )

    
    fig.update_layout(
        title='Sentiment Analysis Results (Bar Plot)',
        margin=dict(t=60, r=20, b=60, l=40),
        font=dict(family='Arial', size=15, color='#d9d7d7'),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
    )

    return save_chart_image(fig)

def generate_pie_chart(results):
    num_neutral = results.get('num_neutral', 0)
    num_positive = results.get('num_positive', 0)
    num_negative = results.get('num_negative', 0)

    labels = ['Neutral', 'Positive', 'Negative']
    values = [num_neutral, num_positive, num_negative]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent',
        marker=dict(colors=['#FFD700', '#90EE90', '#FF6347']),  
    )])

    fig.update_layout(
        title='Sentiment Analysis Results (Pie Plot)',
        margin=dict(t=60, r=20, b=60, l=40),
        font=dict(family='Arial', size=15, color='#d9d7d7'),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
    )

    return save_chart_image(fig)

def save_chart_image(fig):
    # Save the chart to a BytesIO object
    image_stream = BytesIO()
    fig.write_image(image_stream, format="png")
    image_stream.seek(0)

    # Convert the image to base64 for embedding in HTML
    encoded_image = base64.b64encode(image_stream.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"
