import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def mini_lm_encode(df: pd.DataFrame) -> pd.DataFrame:
    '''
    convert the plot_synopsis to a vector using the MiniLM model
    input: df: pd.DataFrame
    output: df_encoded: pd.DataFrame
    title becomes the index
    remaining columns dropped
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df_encoded = df.copy()
    df_encoded['plot_synopsis'] = df_encoded['plot_synopsis'].apply(lambda x : model.encode(x))
    df_index = df_encoded.pop('title')
    df_encoded = df_encoded[['plot_synopsis']]
    df_encoded = pd.DataFrame(np.column_stack(list(zip(*df_encoded.values))))
    df_encoded.index = df_index
    df_encoded.index.name = None
    return df_encoded
