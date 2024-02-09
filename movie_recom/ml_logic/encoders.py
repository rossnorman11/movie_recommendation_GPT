import numpy as np
import pandas as pd
# import embedding with sentence transformer
from sentence_transformers import SentenceTransformer

from movie_recom.ml_logic.data import save_processed_data

# import embedding with bert
from transformers import AutoTokenizer, TFAutoModel


def mini_lm_encode(df: pd.DataFrame) -> pd.DataFrame:
    '''
    convert the plot_synopsis to a vector using the MiniLM model
    input: df: pd.DataFrame
    output: df_encoded: pd.DataFrame
    title becomes the index
    remaining columns dropped
    '''
    # instatiate the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # make copy of df
    df_encoded = df.copy()
    #encode the plot_synopsis
    df_encoded['plot_synopsis'] = df_encoded['plot_synopsis'].apply(lambda x : model.encode(x))
    # remove the title column
    df_index = df_encoded.pop('title')
    # just keep the plot_synopsis, removes the remaining columns
    df_encoded = df_encoded[['plot_synopsis']]
    # plot_synopsis was just one column with a list of numbers, now it's a dataframe with each number in a separate column
    df_encoded = pd.DataFrame(np.column_stack(list(zip(*df_encoded.values))))
    # set the title as the index
    df_encoded.index = df_index
    # remove the index name (title of the index column)
    df_encoded.index.name = None
    save_processed_data(df_encoded, 'data_embedded.csv')
    save_processed_data(df_index, 'data_titelnames.csv')
    return df_encoded

def bert_encode(df: pd.DataFrame) -> pd.DataFrame:
    '''
    convert the plot_synopsis to a vector using bert-tiny
    input: df: pd.DataFrame
    output: df_encoded: pd.DataFrame
    title becomes the index
    remaining columns dropped
    '''

    # instatiate the model
    model_name = "prajjwal1/bert-tiny"
    model = TFAutoModel.from_pretrained(model_name, from_pt = True)
    # instatiate model-specific tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # make copy of df
    df_encoded = df.copy()
    # remove the title column
    df_index = df_encoded.pop('title')
    # just keep the plot_synopsis, removes the remaining columns
    df_encoded = df_encoded[['plot_synopsis']]
    # convert to list for bert
    data_to_embed = df_encoded['plot_synopsis'].tolist()

    # Tokenize the text data
    token_tensor = tokenizer(data_to_embed, padding='max_length', max_length= 500, truncation=True, return_tensors="tf")
    # Create input tensors
    input_tensor = token_tensor['input_ids']
    # Generate embeddings
    prediction = model.predict(input_tensor)
    # Process the embeddings as np
    embedded_data = prediction.last_hidden_state[:, 0, :]
    embedded_data = pd.DataFrame(embedded_data)

    # set the title as the index
    embedded_data.index = df_index
    # remove the index name (title of the index column)
    embedded_data.index.name = None

    return embedded_data
