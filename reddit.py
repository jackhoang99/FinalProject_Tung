import praw
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
)

subreddits = ["EcoFriendly", "BuyItForLife", "ZeroWaste", "Sustainable", "GreenLiving"]
search_query = "eco-friendly"

all_posts_df = pd.DataFrame()

# Fetch posts from each subreddit
for subreddit_name in subreddits:
    print(f"Fetching posts from subreddit: {subreddit_name}")
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(search_query, limit=1000):
        posts.append(
            [
                post.title,
                post.selftext,
                post.score,
                post.num_comments,
                post.created_utc,
                post.author,
                subreddit_name,
            ]
        )

    # Convert to DataFrame
    posts_df = pd.DataFrame(
        posts,
        columns=[
            "Title",
            "Content",
            "Score",
            "Comments",
            "Date",
            "Author",
            "Subreddit",
        ],
    )

    # Append to the main DataFrame
    all_posts_df = pd.concat([all_posts_df, posts_df], ignore_index=True)

# Convert Unix timestamps to datetime
all_posts_df["Date"] = pd.to_datetime(all_posts_df["Date"], unit="s")

# Save to CSV
all_posts_df.to_csv("reddit_data.csv", index=False)

# Load Reddit data
all_posts_df = pd.read_csv("reddit_data.csv")

# Ensure 'Date' is correctly formatted as datetime
all_posts_df["Date"] = pd.to_datetime(all_posts_df["Date"], errors="coerce")

# Set 'Date' as index
all_posts_df.set_index("Date", inplace=True)


# Function to clean text
def clean_text(text):
    if isinstance(text, float):  # Check if the text is a float (e.g., NaN)
        text = ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text_tokens = word_tokenize(text)
    filtered_text = " ".join([word for word in text_tokens if word not in stop_words])
    return filtered_text


stop_words = set(stopwords.words("english"))

# Apply text cleaning with error handling for non-string values
all_posts_df["Cleaned_Content"] = all_posts_df["Content"].apply(lambda x: clean_text(x))

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to analyze sentiment
def analyze_sentiment(text):
    if not isinstance(text, str):
        return "neutral"
    score = sia.polarity_scores(text)
    return (
        "positive"
        if score["compound"] > 0.05
        else "negative" if score["compound"] < -0.05 else "neutral"
    )


# Apply sentiment analysis
all_posts_df["Sentiment"] = all_posts_df["Cleaned_Content"].apply(analyze_sentiment)

# Save data with sentiment analysis
all_posts_df.to_csv("reddit_data_with_sentiment.csv", index=False)

# Keyword Analysis using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
X_reddit = vectorizer.fit_transform(all_posts_df["Cleaned_Content"])
reddit_keywords = vectorizer.get_feature_names_out()
reddit_keyword_df = pd.DataFrame(X_reddit.toarray(), columns=reddit_keywords)
top_keywords_reddit = reddit_keyword_df.sum().sort_values(ascending=False).head(10)

# Aggregate sentiment counts over time and save
posts_sentiment_trend = (
    all_posts_df.resample("M")["Sentiment"].value_counts().unstack().fillna(0)
)

# Plot sentiment trends over time and save
plt.figure(figsize=(10, 5))
posts_sentiment_trend.plot(kind="line", title="Reddit Sentiment Trend Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Posts")
plt.legend(title="Sentiment")
plt.savefig("reddit_sentiment_trend_over_time.png")
plt.close()

keyword = "sustainable"

# Filter posts containing the keyword
posts_keyword_df = all_posts_df[
    all_posts_df["Cleaned_Content"].str.contains(keyword, na=False)
]

# Aggregate and plot trend for specific keyword
keyword_trend = (
    posts_keyword_df.resample("M")["Sentiment"].value_counts().unstack().fillna(0)
)

plt.figure(figsize=(10, 5))
keyword_trend.plot(kind="line", title=f'Trend for "{keyword}" Over Time')
plt.xlabel("Time")
plt.ylabel("Number of Posts")
plt.legend(title="Sentiment")
plt.savefig(f"trend_for_{keyword}_over_time.png")
plt.close()

# Print a confirmation message
print(
    f"Analysis for keyword '{keyword}' and general sentiment trends have been saved successfully."
)
