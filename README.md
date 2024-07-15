Web Scraping and Sentiment Analysis with Flask

This project involves a Flask web application that scrapes web content, analyzes the sentiment, and performs additional insights on the data. It leverages stateoftheart NLP models, data preprocessing techniques, and various machine learning models to provide a comprehensive analysis.

 Features

 Web Scraping: Extracts content from web pages using BeautifulSoup and Selenium.
 Sentiment Analysis: Uses a pretrained BERT model to classify the sentiment of the extracted text.
 Additional Analysis: Provides insights and trends from the scraped data, including categorywise analysis and visualization.
 Flask Web Interface: Simple web interface to input URLs and view results.

 Technologies Used

 Flask: Web framework for Python.
 BeautifulSoup: Library for web scraping.
 Selenium: Automated web browser interaction.
 Transformers (Hugging Face): For loading the pretrained BERT model.
 TextBlob: Simple library for text processing.
 Pandas: Data manipulation and analysis.
 Scikitlearn: Machine learning library for additional analysis.
 Matplotlib: For plotting and visualization.
 NLTK: Natural Language Toolkit for text preprocessing.

 Project Structure

 `app.py`: Main Flask application.
 `templates/`: HTML templates for the web interface.
 `static/`: Static files like CSS and JavaScript.
 `requirements.txt`: Python dependencies.
 `scrapeddata.csv`: CSV file to store scraped and analyzed data.
 `additional_analysis.json`: JSON file to store additional analysis results.
 `best_model_predictions.png`: Plot image showing best model predictions.

 How to Run

1. Clone the repository
    ```bash
    git clone https://github.com/yourusername/webscrapingsentimentanalysis.git
    cd webscrapingsentimentanalysis
    ```

2. Create and activate a virtual environment (optional but recommended)
    ```bash
    python m venv venv
    source venv/bin/activate  On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies
    ```bash
    pip install r requirements.txt
    ```

4. Download NLTK data
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

5. Run the Flask app
    ```bash
    python app.py
    ```

6. Access the web interface
    Open your web browser and go to `http://127.0.0.1:5000`.

 Usage

 Input URL: Enter the URL of the web page you want to scrape and analyze.
 Select Selenium (Optional): Use Selenium for scraping if the page is JavaScriptheavy.
 View Results: Get sentiment analysis, polarity, subjectivity, hashtags, and the scraped content.
 Additional Analysis: View detailed insights and trends from the scraped data.

 Project Details

 Sentiment Analysis
Uses a pretrained DistilBERT model finetuned on the SST2 dataset for sentiment classification.

 Data Preprocessing
 Converts text to lowercase.
 Removes punctuation.
 Tokenizes text.
 Removes stopwords.
 Lemmatizes words.

 Additional Analysis
 Categorizes URLs into predefined categories.
 Vectorizes text data using TFIDF.
 Applies various regression models to analyze sentiment trends.
 Generates visualizations for better understanding.
