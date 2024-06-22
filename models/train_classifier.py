import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """[summary]

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def starting_verb(self, text):
        """checks if the starting work of sentence is verb

        Args:
            text (str): a sentence
        Returns:
            Boolean: return True for verb else False
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """dummy method for verb extractor fit

        Args:
            x ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        return self

    def transform(self, X):
        """transforms sentences

        Args:
            X (numpy.ndarray(str)): array of sentences

        Returns:
            pandas.core.frame.DataFrame: tagged sentences in a pandas dataframe
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """loads data from sqlite databse and returns text and label for modelling

    Args:
        database_filepath (str): path to sqllite database
    Returns:
        numpy.ndarray(str), numpy.ndarray(int), List(str): text, laabels and label column names
    """
    engine = create_engine(f'sqlite:///./{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    Y = df.drop(['id','message', 'original', 'genre'], axis=1).values
    category_names = list(df.drop(['id','message', 'original', 'genre'], axis=1).columns)
    return X, Y, category_names


def tokenize(text):
    """TOkenizes the input text into tokens

    Args:
        text (str): a sentence 

    Returns:
        list(str): list of tokens for the sentence
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """a pipeline model with grid search for text multi output classification

    Returns:
        sklearn.model_selection._search.GridSearchCV: sklearn model for the classification
    """
    parameters = {
        'clf__estimator__bootstrap': (True, False),
        'clf__estimator__criterion': ('gini', 'entropy'),
        'clf__estimator__max_depth': [4,10],
        'clf__estimator__n_estimators': [10, 150]}


    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf',  MultiOutputClassifier(RandomForestClassifier(class_weight='balanced', n_jobs=-1)))
    ])
        
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=10, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluates the model aand prints classification metrics

    Args:
        model (sklearn.model_selection._search.GridSearchCV): a trained model
        X_test (numpy.ndarray(str)): test data
        Y_test (numpy.ndarray): test label
        category_names (list(str)): names of the classes
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """saves model

    Args:
        model (sklearn.model_selection._search.GridSearchCV): sklearn grid search model
        model_filepath (str): model save location
    """
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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