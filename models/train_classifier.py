import sys
import re
import pickle
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report




def load_data(database_filepath):
    '''
    load data from the sql database
    
    INPUT:
    database_filepath - a string of the path to the database
    
    OUTPUT:
    X - pandas Series of messages
    Y - subset of df containing the category columns data
    category_names - a list of the category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('ResponseMessages', engine)
    
    X = df['message']
    y = df.iloc[:,5:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    lemm = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [w for w in words if w not in string.punctuation]
    words = [w for w in words if w not in stopwords.words("english")]
    words = [lemm.lemmatize(word).lower().strip() for word in words]
    
    return ' '.join(words)


def build_model():
    '''
    Build a ML pipeline with a set of parameters to use in grid search.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
                  'vect__max_features': (5000, 10000),
                  'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 4]}
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Genereate the classification report for the model.
    '''
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred.columns = category_names
    
    reports = []
    for category in category_names:
        report = classification_report(Y_test[category], y_pred[category])
        reports.append(report)
        
        print("Classification report for: ", category)
        print(report)
        


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()