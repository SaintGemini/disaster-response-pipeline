# Disaster Response Pipeline Project

### Project Overview
The purpose of this project was to help build fundamentals in building ETL pipelines, ML pipelines, and
NLP data clealing libraries. Using real world disaster response messages the ML model will classify each message
it is given under certain disaster categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Files in Repository
* app 
<br>| - templates
|| - master.html                # home page of web app
|| - go.html                    # results page showing predicted classification

* data
| - disaster_categories.csv     # data to be processed
| - disaster_messages.csv       # data to be processed
| - process_data.py             # python script that cleans data and saves to SQL database
| - DisasterResponse.db         # SQL database genereated from process_data.py

* models
| - train_classifier.py         # python script that trains ML model and saves to pickle file 
| - classifier.pkl              # ML model saved as pickle file
