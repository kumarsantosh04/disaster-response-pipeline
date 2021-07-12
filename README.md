# Disaster Response Pipeline Project
## Classifies message from a web UI into 36 different categories. 
Loads messages and message categories data, trains a model on it and uses that model to classify a message into 36 categories on a web UI.

### Install requirements:
    
Run the following commands in the project's root directory to install dependencies
    
~~~~
pip install -r requirements.txt
~~~~

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ~~~~
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ~~~~
    - To run ML pipeline that trains classifier and saves
        ~~~~
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ~~~~

2. Run the following command in the app's directory to run your web app.
    ~~~~
    python run.py
    ~~~~

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/

### Project Structure:

~~~~
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
- requirements.txt
~~~~

### Project Components:

There are three components in this project.

1. **ETL Pipeline**
    
    ```In a Python script, `process_data.py`, a data cleaning pipeline that:```

    * Loads the messages and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores it in a SQLite database

2. **ML Pipeline**
    
    ```In a Python script, `train_classifier.py`, a machine learning pipeline that:```

    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file
 
 3. **App(WebUI)**

### Acknowledgement
Udacity for providing the WebUI template code.
