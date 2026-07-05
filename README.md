Universal Review Analyzer  
A Streamlit dashboard for analyzing customer feedback across any dataset or typed reviews.

 Features
 
CSV Upload → Upload datasets with reviews, ratings, and timestamps.

Instant Text Classification → Type a review directly and get Positive/Negative/Neutral sentiment instantly.

Dataset Overview → Preview sample reviews, total count, and average rating.

Sentiment Analysis → VADER scoring with bar chart distribution and inline explanations.

Word Cloud → Visualize frequent terms with explanations of meaning.

Trend Analysis → Track sentiment changes over time.

Top Reviews → Highlight most positive and most negative reviews with reasoning.

 Design
 
Dark theme with teal/orange accents for a modern, sleek look.

Responsive layout with sidebar navigation.

 Tech

 
Built with Streamlit.

Sentiment analysis powered by NLTK VADER.

Smart CSV encoding fallback and auto‑detection of review/rating/timestamp columns.

CSV auto-detection supports common review exports including `text`, `tweet_text`, `full_text`, `comment`, `feedback`, and `body` columns from tools such as Xquik API.

 Run Locally

 
bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
streamlit run universal_review_analyzer.py
Opens at: http://localhost:8501
