import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load the two csv files in their own pandas DataFrame, then merge them based on common id
    
    INPUT:
    messages_filepath - a string with the file path to messages.csv
    categories_filepath - a string with the file path to categories.csv
    
    OUTPUT:
    df - a single dataframe with the two csv files merged on common id
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    '''
    Steps to clean the data:
    
    1) split the categories column into separate values so each value can become its own column
    2) rename the new columns created from step 1
    3) convert values in categories columns to integer
    4) remove any duplicates or rows with nan values in all categories
    
    INPUT:
    df - dataframe with data to be cleaned
    
    OUTPUT:
    df - dataframe with clean data
    
    '''
    # split all the categories and store in separate dataframe
    categories = df.categories.str.split(';', expand=True)
    
    # grab first row of categories dataframe
    first_row = categories.iloc[0]
    
    category_colnames = []
    # grab only the column name from each cell
    for col in first_row:
        category_colnames.append(col[:-2])
        
    categories.columns = category_colnames
    
    # convert values to 0 or 1
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = pd.to_numeric(categories[col], downcast='integer')
        
    # replace categories column in df with the new category columns
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates and rows with missing values in all of the category columns
    df.drop_duplicates(inplace=True)
    df = df.dropna(axis=0, subset=category_colnames)
    
    # converts related column to binary
    df['related'] = np.where((df['related'] == 2), 1, df['related'])
    
    return df

def save_data(df, database_filename):
    '''
    Save data to a SQLite database.

    INPUT:
    df - dataframe to save in database
    database_filename - string to use as database name
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('ResponseMessages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()