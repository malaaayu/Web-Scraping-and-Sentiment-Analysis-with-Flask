import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import re
import json
import matplotlib.pyplot as plt

def run_additional_analysis():
    data = pd.read_csv('scrapeddata.csv', header=0)

    expected_columns = ['url', 'sentiment', 'polarity', 'subjectivity', 'hashtags', 'insights']
    if not all(col in data.columns for col in expected_columns):
        data.columns = expected_columns

    data['insights'] = data['insights'].fillna('')

    categories = {
        'Electronics': ['electronics', 'gadgets', 'phones', 'computers'],
        'Fashion': ['fashion', 'clothing', 'apparel', 'accessories'],
        'Books': ['books', 'reading', 'novels', 'literature'],
        'Home & Garden': ['garden', 'decor', 'furniture'],
        'Health & Fitness': ['health', 'fitness', 'exercise', 'wellness'],
        'E-commerce': ['amazon', 'flipkart', 'ebay', 'shopping', 'shop'],
        'News': ['timesofindia', 'economictimes','indiatoday', 'news', 'bbc', 'cnn'],
        'Education': ['edu', 'university', 'college', 'school', 'course', 'questions'],
        'Storage': ['drive', 'cloud storage']
    }

    def categorize_url(url):
        url = url.lower()
        for category, patterns in categories.items():
            for pattern in patterns:
                if re.search(pattern, url):
                    return category
        return 'Other'

    data['category'] = data['url'].apply(categorize_url)

    if 'insights' not in data.columns or 'sentiment' not in data.columns:
        raise KeyError("The 'insights' or 'sentiment' column is missing in the dataset.")

    X = data['insights']
    y = data['sentiment']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Machine': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor()
    }

    model_metrics = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_metrics[model_name] = {'MSE': mse, 'R2': r2}

    best_model = max(model_metrics, key=lambda x: model_metrics[x]['R2'])
    best_model_instance = models[best_model]
    best_model_instance.fit(X_train, y_train)
    y_pred_best = best_model_instance.predict(X_test)

    # Plotting best model predictions versus actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, color='blue', label='Predictions')
    plt.plot(y_test, y_test, color='red', linestyle='--', label='Actual')
    plt.title(f'Actual vs Predicted Values for {best_model}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as an image file
    plot_filename = 'best_model_predictions.png'
    plt.savefig(plot_filename)

    # Additional analysis for e-commerce, news, and education categories
    ecommerce_data = data[data['category'] == 'E-commerce']
    ecommerce_category_counts = ecommerce_data['category'].value_counts().to_dict()
    ecommerce_trending_hashtags = ecommerce_data['hashtags'].dropna().tolist()
    ecommerce_complete_data = ecommerce_data.to_json(orient='records')

    news_data = data[data['category'] == 'News']
    news_category_counts = news_data['category'].value_counts().to_dict()
    news_trending_hashtags = news_data['hashtags'].dropna().tolist()
    news_complete_data = news_data.to_json(orient='records')

    education_data = data[data['category'] == 'Education']
    education_category_counts = education_data['category'].value_counts().to_dict()
    education_trending_hashtags = education_data['hashtags'].dropna().tolist()
    education_complete_data = education_data.to_json(orient='records')

    # Filter ecommerce data (e.g., Flipkart)
    ecommerce_data_flipkart = data[data['url'].str.contains('flipkart', case=False, na=False)]

    # Display the category-wise count of ecommerce data
    ecommerce_category_counts_flipkart = ecommerce_data_flipkart['category'].value_counts().to_dict()

    # Assuming 'hashtags' is the column containing trending hashtags
    trending_hashtags_flipkart = ecommerce_data_flipkart['hashtags']

    # Extract insights of ecommerce URLs
    insights_d_flipkart = ecommerce_data_flipkart['insights']

    # Define a list of product names to search for
    product_names = ["Samsung", "Apple", "Sony", "Oneplus", "Smartwatches", "Smartphone", "Grocery", "Fashion"]

    # Define a regular expression pattern to match product names
    product_name_pattern = '|'.join(product_names)

    # Define a regular expression pattern to match product information
    product_info_pattern = r'(?i)(?:\d+\s*GB)?\s*(?:\d+\s*[xX])?\s*(?:\d+\s*hours)?\s*[A-Za-z\s]+\d+(?:\.\d+)?(?:\s*[A-Za-z\s]+)*'

    # Create a dictionary to store product information
    product_info_dict = {}

    # Iterate over each insight in the 'insights' column
    for insight in insights_d_flipkart.dropna():
        # Find all matches of the product name pattern
        product_name_matches = re.findall(product_name_pattern, insight, flags=re.IGNORECASE)
        # Find all matches of the product information pattern
        product_info_matches = re.findall(product_info_pattern, insight)
        # If both product name and product information are found, store them together
        if product_name_matches and product_info_matches:
            for product_name in product_name_matches:
                product_info_dict.setdefault(product_name, []).extend(product_info_matches)

    # Additional analysis for storage and other categories
    storage_data = data[data['category'] == 'Storage']
    storage_category_counts = storage_data['category'].value_counts().to_dict()
    storage_trending_hashtags = storage_data['hashtags'].dropna().tolist()
    storage_complete_data = storage_data.to_json(orient='records')

    other_data = data[data['category'] == 'Other']
    other_category_counts = other_data['category'].value_counts().to_dict()
    other_trending_hashtags = other_data['hashtags'].dropna().tolist()
    other_complete_data = other_data.to_json(orient='records')

    additional_analysis_data = {
        "best_model": best_model,
        "model_metrics": model_metrics,
        "ecommerce_category_counts": ecommerce_category_counts,
        "news_category_counts": news_category_counts,
        "education_category_counts": education_category_counts,
        "ecommerce_trending_hashtags": ecommerce_trending_hashtags,
        "news_trending_hashtags": news_trending_hashtags,
        "education_trending_hashtags": education_trending_hashtags,
        "ecommerce_complete_data": json.loads(ecommerce_complete_data),
        "news_complete_data": json.loads(news_complete_data),
        "education_complete_data": json.loads(education_complete_data),
        "ecommerce_category_counts_flipkart": ecommerce_category_counts_flipkart,
        "trending_hashtags_flipkart": trending_hashtags_flipkart.tolist(),
        "insights_ecommerce_flipkart": insights_d_flipkart.tolist(),
        "product_info_ecommerce_flipkart": product_info_dict,
        "plot_filename": plot_filename,
        "storage_category_counts": storage_category_counts,
        "storage_trending_hashtags": storage_trending_hashtags,
        "storage_complete_data": json.loads(storage_complete_data),
        "other_category_counts": other_category_counts,
        "other_trending_hashtags": other_trending_hashtags,
        "other_complete_data": json.loads(other_complete_data)
    }

    with open('additional_analysis.json', 'w') as f:
        json.dump(additional_analysis_data, f)

    print("Storage Category Counts:", storage_category_counts)
    print("Storage Trending Hashtags:", storage_trending_hashtags)
    print("Storage Complete Data:", json.loads(storage_complete_data))
    print("Other Category Counts:", other_category_counts)
    print("Other Trending Hashtags:", other_trending_hashtags)
    print("Other Complete Data:", json.loads(other_complete_data))

    return json.dumps(additional_analysis_data)

# Run the function
run_additional_analysis()
