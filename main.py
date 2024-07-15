from flask import Flask, render_template, request
import csv
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import nltk
import json
nltk.download('wordnet')


from textblob import TextBlob


app = Flask(__name__)


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def preprocess_text(text):
   text = text.lower()
   text = ''.join([c for c in text if c not in string.punctuation])
   tokens = word_tokenize(text)
   stop_words = stopwords.words('english')
   filtered_words = [w for w in tokens if w not in stop_words]
   lemmatizer = WordNetLemmatizer()
   lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
   return ' '.join(lemmatized_words)


def scrape_and_analyze(url, use_selenium=False):
   if use_selenium:
       try:
           driver = webdriver.Chrome('/path/to/chromedriver')
           driver.get(url)
           time.sleep(2)
           soup = BeautifulSoup(driver.page_source, 'html.parser')
           driver.quit()
       except Exception as e:
           print(f"Selenium error: {e}")
           return None
   else:
       try:
           response = requests.get(url)
           if response.status_code != 200:
               return None
           soup = BeautifulSoup(response.text, 'html.parser')
       except Exception as e:
           print(f"Request error: {e}")
           return None


   scraped_content = " ".join([p.get_text() for p in soup.find_all('p')])
   processed_text = preprocess_text(scraped_content)


   encoded_text = tokenizer(processed_text, return_tensors='pt')
   output = model(**encoded_text)
   prediction = int(torch.argmax(output.logits))
   sentiment = 'positive' if prediction > 0 else 'negative'


   text_blob = TextBlob(scraped_content)
   polarity = text_blob.polarity
   subjectivity = text_blob.subjectivity


   hashtags = [a.text for a in soup.find_all('a', href=lambda href: href and href.startswith('#'))]


   data_to_save = [url, sentiment, polarity, subjectivity, hashtags, scraped_content]
   with open('scrapeddata.csv', 'a', newline='', encoding='utf-8') as csvfile:
       csv_writer = csv.writer(csvfile)
       csv_writer.writerow(data_to_save)


   return sentiment, polarity, subjectivity, hashtags, scraped_content


@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'POST':
       url = request.form['url']
       use_selenium = request.form.get('use_selenium', False)


       sentiment, polarity, subjectivity, hashtags, scraped_content = scrape_and_analyze(url, use_selenium)


       if sentiment:
           from ss import run_additional_analysis
           additional_analysis_data = json.loads(run_additional_analysis())
           return render_template('result.html', url=url, sentiment=sentiment, polarity=polarity,
                                  subjectivity=subjectivity, hashtags=hashtags, scraped_content=scraped_content,
                                  additional_analysis_data=additional_analysis_data)
       else:
           return render_template('result.html', url=url, error="Failed to analyze the webpage content. Please try again.")


   return render_template('index.html')


if __name__ == '__main__':
   app.run(debug=True)

