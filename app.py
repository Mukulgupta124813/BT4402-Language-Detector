from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

codel = {"English":0,
"French":1,
"Spanish":2,
"Portugeese":3,
"Italian":4,
"Russian":5,
"Sweedish":6,
"Malayalam":7,
"Dutch":8,
"Arabic":9,
"Turkish":10,
"German":11,
"Tamil":12,
"Danish":13,
"Kannada":14,
"Greek":15,
"Hindi":16}
def getcodel(n) :
    for x , y in codel.items() :
        if n == y :
            return x

# input = "വളരെ പെട്ടെന്നു"
with open('count.pickle', 'rb') as handle:
    count = pickle.load(handle)
with open('MultinomialNB.sav', 'rb') as handle:
    language_detect_model = pickle.load(handle)
# x = count.transform([input])
# print(getcodel(language_detect_model.predict(x)))

@app.route('/', methods=['GET','POST'])
def Detector():
    input = request.args.get('Input')
    x = count.transform([input])
    res = {'Data': input, 'Output':getcodel(language_detect_model.predict(x))}
    return jsonify(res)

if __name__ == '__main__':
   app.run(host='0.0.0.0')