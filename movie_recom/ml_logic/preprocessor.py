import numpy as np
import pandas as pd
from colorama import Fore, Style

def shorten_synopsis(max_len: int, df: pd.DataFrame) -> pd.DataFrame:
    '''To Do: shorten long synopsis to max_len words'''
    df['plot_word_count'] = df['plot_synopsis'].apply(lambda x: len(x.split()))
    df_output = df[df['plot_word_count'] < max_len].copy()
    df_output = df_output.drop(columns = 'plot_word_count', axis = 1)
    return df_output
