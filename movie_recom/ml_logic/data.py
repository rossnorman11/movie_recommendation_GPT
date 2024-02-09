import pandas as pd
from colorama import Fore, Style
from movie_recom.params import *
from pathlib import Path

def get_raw_data() -> pd.DataFrame:
    '''loads the raw data from the mpst_full_data.csv file from hard drive'''
    # Get the parent folder of the current file (goes up 2 levels)
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # path and filename for data in movie_recommendation_GPT folder
    raw_data_path = Path(parent_folder_path).joinpath("raw_data/mpst_full_data.csv")
    # get the data
    df = pd.read_csv(raw_data_path)
    print("✅ get_data() done \n")
    return df

def save_embedded_data(df: pd.DataFrame) -> None:
    '''saves the embedded data to the hard drive'''
    # Get the parent folder of the current file (goes up 2 levels)
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Define the path and foldername to save the data
    embedded_data_path = Path(parent_folder_path).joinpath("processed_data/data_embedded.csv")
    # save our data
    df.to_csv(embedded_data_path, index=True)
    print("✅ save_data() done \n")

def get_embedded_data() -> pd.DataFrame:
    '''loads the embedded data from the data_embedded.csv file from hard drive'''
    # Get the parent folder of the current file (goes up 2 levels)
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # path and filename for data in movie_recommendation_GPT folder
    embedded_data_path = Path(parent_folder_path).joinpath("processed_data/data_embedded.csv")
    # get the data
    df_embedded = pd.read_csv(embedded_data_path, index_col=0)
    print("✅ get_embedded_data() done \n")
    return df_embedded

if __name__ == '__main__':
    pass
