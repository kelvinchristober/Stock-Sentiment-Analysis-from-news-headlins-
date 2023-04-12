from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)

#model = pickle.load(open('model2.pkl', 'rb'))
#cv = pickle.load(open('transform.pkl', 'rb'))

#df = pd.read_csv('Data.csv', encoding='ISO-8859-1')
#rain = df[df['Date'] < '20150101']

## Renaming column names for ease of access
#list1= [i for i in range(25)]
#new_Index=[str(i) for i in list1]
#data.columns= new_Index

#Implementing bag of words
#train_data = cv.transform(headlines)

## Implementing Random Forest Classifier
#rfc = RandomForestClassifier(n_estimators=200, criterion='entropy')
#rfc.fit(train_data, train['Label'])

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    test_data = request.form['test_data'].lower()

    processed_data = ' '.join([word for word in test_data.split() if word not in stop_words])

    sia = SentimentIntensityAnalyzer()
    dd = sia.polarity_scores(text=processed_data)
    compound = round((1 + dd['compound'])/2, 2)
    
    if compound > 0.5:
        compound1 = "1"
    elif compound < 0.5:
        compound1 = "0"
    else:
        compound1 = "0"

    # Categorize sentiment score
    if compound > 0.5:
        sentiment = "positive"
    elif compound < 0.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return render_template('form.html', final=compound1, test_data=test_data, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
