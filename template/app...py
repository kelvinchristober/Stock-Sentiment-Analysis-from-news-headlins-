from flask import Flask, request, render_template
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, template_folder="template")

model = pickle.load(open('model2.pkl', 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('Data.csv', encoding='ISO-8859-1')
train = df[df['Date'] < '20150101']

# Renaming column names for ease of access
data = train.iloc[:, 2:27]
data.columns = [str(i) for i in range(25)]

# Convertng headlines to lower case
data = data.apply(lambda x: x.str.lower())

# Joining headline columns into a single string
data['headlines'] = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Implementing bag of words
train_data = cv.transform(data['headlines'])

# Implementing Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy')
rfc.fit(train_data, train['Label'])

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    test_data = request.form['text1'].lower()
    processed_data = [word for word in test_data.split() if word not in stop_words]
    transformed_data = cv.transform(processed_data).toarray()
    prediction = rfc.predict(transformed_data)
    return render_template('form.html', final=prediction, test_data=test_data)

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5002, threaded=True)