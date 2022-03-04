import json
import plotly
import pandas as pd
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    lemm = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [w for w in words if w not in string.punctuation]
    words = [w for w in words if w not in stopwords.words("english")]
    words = [lemm.lemmatize(word).lower().strip() for word in words]
    
    return ' '.join(words)

def sort_distribution(col_names, col_vals):
    '''
    Helper function for data for data visualization in ascending order.
    
    INPUT:
    col_names - a list of strings, must be in the same order as col_vals
    col_vals - a list of values (the sum count of occurances of col_names)
    
    OUTPUT:
    col_names - a sorted list in ascending order of the column names
    col_vals - a sorted list in ascendin order of the column values
    '''
    lst = []
    # create a list of pairs so they will sort in correct order
    for name, val in zip(col_names, col_vals):
        lst.append((name, val))
    # sort list by col_val in ascending order
    lst.sort(key=lambda x:x[1])
    
    col_names = []
    col_vals = []
    
    # extract data into separate lists
    for item in lst:
        col_names.append(item[0])
        col_vals.append(item[1])
        
    return col_names, col_vals

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('ResponseMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    genre_names, genre_counts = sort_distribution(genre_names, genre_counts)
    
    category_names = df.iloc[:,4:].columns
    category_vals = (df.iloc[:,4:] != 0).sum().values
    
    category_names, category_vals = sort_distribution(category_names, category_vals)
        
   
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
            # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            # GRAPH 2 - category value sum graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_vals
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results  safsah
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()