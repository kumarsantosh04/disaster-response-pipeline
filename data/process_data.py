import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """loads message and categories data and join them

    Args:
        messages_filepath (str): csv filepath of message file
        categories_filepath (str): csv filepath of categories file

    Returns:
        pandas.core.frame.DataFrame: joined data as pandas dataframe
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = pd.merge(messages_df, categories_df, on=['id'], how='inner')
    return df


def clean_data(df):
    """clean categories of data and drops duplicates

    Args:
        df (pandas.core.frame.DataFrame:): pandas dataframe containing categories

    Returns:
        pandas.core.frame.DataFrame: cleaned data
    """
    #Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    category_colnames = [r[:-2] for r in row.values[0]]
    categories.columns = category_colnames


    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #Replace categories column in df with new category columns.
    df.drop("categories", axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)


    #Remove duplicates.
    # check number of duplicates
    sum(df.duplicated())
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    sum(df.duplicated())

    return df


def save_data(df, database_filename):
    """saves data to sqlite database

    Args:
        df (pandas.core.frame.DataFrame:): dataframe to be saved
        database_filename (str): database file path
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)  


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