import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import sleep
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download("punkt")


def soup2list(src, list_, attr=None):
    if attr:
        for val in src:
            list_.append(val[attr])
    else:
        for val in src:
            list_.append(val.get_text())


# List of eco-friendly companies
companies = ["earthhero.com", "thegreencompany.online", "ecoleafproducts.co.uk"]

# Initialize an empty DataFrame to store all reviews
all_review_data = pd.DataFrame(
    columns=[
        "Company",
        "Username",
        "Total reviews",
        "Location",
        "Date",
        "Content",
        "Rating",
    ]
)

# Define the page range to scrape
from_page = 1
to_page = 6

for company in companies:
    users = []
    userReviewNum = []
    ratings = []
    locations = []
    dates = []
    reviews = []

    print(f"Scraping reviews for company: {company}")

    for i in range(from_page, to_page + 1):
        url = f"https://www.trustpilot.com/review/{company}?page={i}"
        result = requests.get(url)
        soup = BeautifulSoup(result.content, "html.parser")

        soup2list(
            soup.find_all(
                "span",
                {
                    "class",
                    "typography_heading-xxs__QKBS8 typography_appearance-default__AAY17",
                },
            ),
            users,
        )
        soup2list(
            soup.find_all(
                "div",
                {
                    "class",
                    "typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_detailsIcon__Fo_ua",
                },
            ),
            locations,
        )
        soup2list(
            soup.find_all(
                "span",
                {
                    "class",
                    "typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l",
                },
            ),
            userReviewNum,
        )
        soup2list(soup.find_all("div", {"class", "styles_reviewHeader__iU9Px"}), dates)
        soup2list(
            soup.find_all("div", {"class", "styles_reviewHeader__iU9Px"}),
            ratings,
            attr="data-service-review-rating",
        )
        soup2list(
            soup.find_all("div", {"class", "styles_reviewContent__0Q2Tg"}), reviews
        )

        # Print lengths of lists for debugging
        print(
            f"Page {i} - Users: {len(users)}, Locations: {len(locations)}, UserReviewNum: {len(userReviewNum)}, Dates: {len(dates)}, Ratings: {len(ratings)}, Reviews: {len(reviews)}"
        )

        # To avoid throttling
        sleep(1)

    # Ensure all lists have the same length by padding shorter lists
    max_length = max(
        len(users),
        len(userReviewNum),
        len(ratings),
        len(locations),
        len(dates),
        len(reviews),
    )

    users += [""] * (max_length - len(users))
    userReviewNum += [""] * (max_length - len(userReviewNum))
    ratings += [""] * (max_length - len(ratings))
    locations += [""] * (max_length - len(locations))
    dates += [""] * (max_length - len(dates))
    reviews += [""] * (max_length - len(reviews))

    # Create a DataFrame for the current company
    review_data = pd.DataFrame(
        {
            "Company": company,
            "Username": users,
            "Total reviews": userReviewNum,
            "Location": locations,
            "Date": dates,
            "Content": reviews,
            "Rating": ratings,
        }
    )

    # Append the current company reviews to the all_review_data DataFrame
    all_review_data = pd.concat([all_review_data, review_data], ignore_index=True)

# Save the combined data to a CSV file
all_review_data.to_csv("all_reviews.csv", index=False)

print("Scraping completed. Data saved to all_reviews.csv")

# Load and clean the data
reviews_df = pd.read_csv("all_reviews.csv")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(
        r"[^a-zA-Z0-9\s]", "", text
    )  # Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()


# Fill missing values in 'Content' with an empty string
reviews_df["Content"] = reviews_df["Content"].fillna("")
reviews_df["Rating"] = reviews_df["Rating"].fillna(reviews_df["Rating"].median())

# Apply the cleaning function
reviews_df["Cleaned_Content"] = reviews_df["Content"].apply(clean_text)

# Initialize NLTK VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] > 0.05:
        return "positive"
    elif scores["compound"] < -0.05:
        return "negative"
    else:
        return "neutral"


# Apply sentiment analysis
reviews_df["Sentiment"] = reviews_df["Cleaned_Content"].apply(analyze_sentiment)

# Save the data with sentiment analysis
reviews_df.to_csv("reviews_with_sentiment.csv", index=False)
print("Sentiment analysis completed and saved to reviews_with_sentiment.csv")

# Preprocess text for topic modeling
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Apply preprocessing
reviews_df["Processed_Content"] = reviews_df["Cleaned_Content"].apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
dtm = vectorizer.fit_transform(reviews_df["Processed_Content"])

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -no_top_words - 1 : -1]]
            )
        )


# Display the topics
no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

# Summarize sentiment analysis results
sentiment_summary = reviews_df["Sentiment"].value_counts(normalize=True) * 100
print("Sentiment Summary:")
print(sentiment_summary)

# Save sentiment summary
sentiment_summary.to_csv("sentiment_summary.csv", header=True)
print("Sentiment summary saved to sentiment_summary.csv")

# Save topics to a file
topics = []
for topic_idx, topic in enumerate(lda.components_):
    topics.append(
        [
            vectorizer.get_feature_names_out()[i]
            for i in topic.argsort()[: -no_top_words - 1 : -1]
        ]
    )

topics_df = pd.DataFrame(topics, columns=[f"Word {i+1}" for i in range(no_top_words)])
topics_df.to_csv("topics.csv", index=False)
print("Topics saved to topics.csv")


# Combine sentiment and rating categories
def categorize_rating(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


reviews_df["Rating_Category"] = reviews_df["Rating"].apply(categorize_rating)

# Cross-tabulation of Sentiment and Rating Category
crosstab = (
    pd.crosstab(
        reviews_df["Sentiment"], reviews_df["Rating_Category"], normalize="index"
    )
    * 100
)

# Plotting the cross-tabulation
sns.heatmap(crosstab, annot=True, cmap="coolwarm", cbar=True)
plt.title("Sentiment vs. Rating Category")
plt.xlabel("Rating Category")
plt.ylabel("Sentiment")
plt.show()

# Calculate the correlation matrix
reviews_df["Sentiment_Score"] = reviews_df["Sentiment"].replace(
    {"positive": 1, "neutral": 0, "negative": -1}
)
correlation_matrix = reviews_df[["Rating", "Sentiment_Score"]].corr()

# Plotting the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation between Rating and Sentiment")
plt.show()
