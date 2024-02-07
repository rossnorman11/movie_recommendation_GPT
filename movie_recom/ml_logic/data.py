import pandas as pd
from colorama import Fore, Style
from movie_recom.params import *

def get_raw_data() -> pd.DataFrame:
    # Load data
    df = pd.read_csv('../../raw_data/mpst_full_data.csv')
    print("✅ get_data() done \n")
    return df

def save_embedded_data(df: pd.DataFrame) -> None:
    # Save data
    df.to_csv('../../processed_data/data_embedded.csv', index=True)
    print("✅ save_data() done \n")

def get_embedded_data() -> pd.DataFrame:
    # Load data
    df_embedded = pd.read_csv('../../processed_data/data_embedded.csv', index_col=0)
    print("✅ get_embedded_data() done \n")
    return df_embedded
