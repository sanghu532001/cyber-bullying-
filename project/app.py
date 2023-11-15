from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.impute import SimpleImputer 
import re 
import nltk 
nltk.download('stopwords') 
from nltk.util import pr 
stemmer=nltk.SnowballStemmer('english') 
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import string 

app = Flask(__name__)

# Load and clean the dataset
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({0: "Hate Speech Detection", 1: "offensive language detected", 3: "no hate and offensive speech"})
df['labels'] = df['labels'].fillna("no hate and offensive speech")
df = df[['tweet','labels']]
df['labels'] = df['labels'].fillna(0)
df['labels'] = df['labels'].astype(str)

def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('https?://\S+|www\. \S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)

# Train the machine learning model
x = np.array(df["tweet"])
y = np.array(df["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)



# Define the web app routes
@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        tweet = request.form['input-text']
        if tweet.strip():
            clean_tweet = clean(tweet)
            tweet_vec = cv.transform([clean_tweet]).toarray()
            prediction = clf.predict(tweet_vec)
            print(prediction)
            return render_template('index.html', prediction=prediction)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
