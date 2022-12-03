
#region imports
import pandas as pd
import numpy as np
import random
import re
from nltk import word_tokenize
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import set_config
set_config(display="diagram")
rand_state=100
RANDOMSEED = 100
DISPLAY_WIDTH = 400
DISPLAYMAX_COLUMNS = 25
#endregion
#region settings
random.seed(RANDOMSEED)
pd.set_option('display.width', DISPLAY_WIDTH)
pd.set_option('display.max_columns', DISPLAYMAX_COLUMNS)
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')

#endregion


def tokenize(text):
    '''this method does the following
    1. normalizing all the words to lower size
    2. removes punctuations
    3. splits the words
    4. removes the stopwords like am,is,have,you,...
    5. lammetizes the words for example running-->run
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    # normalize case and remove punctuation
    tokens = word_tokenize(text)    # tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    # lemmatize andremove stop words
    return tokens


def prep_data(text,method=CountVectorizer):
    '''
    this method counts either counts the words 
    in sentences (CountVectorizer) or wights them 
    based on their importance in the sentence 
    and entire data(TfidfVectorizer):
    '''
    count_vector = method(tokenizer=tokenize)
    count_vector.fit(text)
    doc_array = count_vector.transform(text).toarray()
    frequency_matrix_count = pd.DataFrame(doc_array, columns=count_vector.get_feature_names_out())
    return frequency_matrix_count


def tweak_details(df_all_details):
    return (#1- major cleanups
    df_all_details  
        .query("status in ['Released']")
        .pipe(lambda df_:df_.replace("none",np.NaN))
        .assign(revenue=lambda df_:df_.revenue.replace(0,np.NAN),
            genres=lambda df_:df_.genres.fillna("none"),
            runtime=lambda df_:df_.runtime.fillna(-999),
            # original_language=lambda df_:df_.original_language.fillna("none"),
            day_of_week_temp=lambda df_:pd.to_datetime(df_.release_date,dayfirst=True),
            day_of_week=lambda f_:f_.day_of_week_temp.apply( lambda d:d.weekday()),
            year=lambda df_:pd.to_datetime(df_.release_date,dayfirst=True).dt.year,
            age=lambda df_:(2025-pd.to_datetime(df_.release_date,dayfirst=True).dt.year).fillna(-999),
            month=lambda df_:(pd.to_datetime(df_.release_date,dayfirst=True).dt.month),
            sin_month=lambda df_:(np.sin(2*np.pi*(df_.month-1)/12)).fillna(-999),
            cos_month=lambda df_:(np.cos(2*np.pi*(df_.month-1)/12)).fillna(-999),
            popularity=lambda df_:df_.popularity.astype("str").apply(lambda x:x.replace(",","")).astype("float") ,
            original_language=lambda df_:df_.original_language.apply(lambda x:1 if x=="en" else 0),
            production_countries=lambda df_:df_.production_countries.apply(lambda x:1 if x=="United States of America" else 0),
            spoken_languages=lambda df_:df_.spoken_languages.apply(lambda x:1 if x=="English" else 0),
      
            )
        
        .dropna(subset="revenue")
        .query("revenue > 0")
        .reset_index(drop=True)
        .pipe(lambda df_:pd.concat([df_,
                                prep_data(df_.genres)],axis=1))
        .drop(columns=["imdb_id","original_title",
                    "overview","status","tagline","title","vote_average","vote_count",
                    "production_companies","release_date","day_of_week_temp","month"])
    
    )
    
def fame_func(data_awards_cleaned,director,dt_,stat=0):

    try: 
        if stat==0:
            return data_awards_cleaned.loc[director][data_awards_cleaned.loc[director].index<=dt_].tail(1).nominated_cumsum.values[0]
        else:
            return data_awards_cleaned.loc[director][data_awards_cleaned.loc[director].index<=dt_].tail(1).won_cumsum.values[0]
            
    except:
        return np.NaN

def tweak_cast(df_all_casting,df):
    return(#2- extracting actor_weights
            df_all_casting 
                .melt(id_vars="id",value_vars=["actor1_name","actor2_name","actor3_name","actor4_name","actor5_name"])
                .sort_values(by=["id","variable"])
                .merge(df[["id","year"]],on="id",how="left")
                .replace("none",np.NaN)
                .dropna()
                .assign(unit=1)
                .sort_values(by=["value","year"])
                .assign(actor_freq=lambda df_:df_.groupby("value").cumsum()["unit"])
                .pipe(lambda df_:pd.pivot(df_,values="actor_freq",columns="variable",index="id"))
                .reset_index()
            )
    
def tweak_director(df_all_casting,df):
    return (#2- extracting director_weights
                    df_all_casting 
                    .melt(id_vars="id",value_vars=["director_name"])
                    .sort_values(by=["id","variable"])
                    .merge(df[["id","year"]],on="id",how="left")
                    .replace("NaN",np.NaN)
                    .replace("none",np.NaN)
                    .dropna()
                    .assign(unit=1)
                    .sort_values(by=["value","year"])
                    .assign(director_freq=lambda df_:df_.groupby("value").cumsum()["unit"])
                    .drop(columns=["variable","value","year","unit"])
                    .merge(df_all_casting[["id","director_name"]],on="id",how="left")
    )
      
def data_awards_cleaned_func(df_awards):
    return (
        df_awards
                .assign(outcome=lambda df_:df_["outcome"].replace(["2nd place", "3rd place"], "Won"))
                .groupby(["director_name", "year","outcome"])
                .count()
                .unstack()
                .fillna(0)
                .reset_index()
                .iloc[:,0:4]
                .pipe(lambda df_:pd.DataFrame(df_.values,columns=["director_name","year","nominated","won"]))
                .assign(nominated_cumsum = lambda df_: df_.groupby("director_name")["nominated"].transform(pd.Series.cumsum),
                    won_cumsum = lambda df_: df_.groupby("director_name")["won"].transform(pd.Series.cumsum),
                    )
                .set_index(["director_name","year"])
            )
    
def tweak_data(df_all_casting,df_all_details,df_awards):
    df=tweak_details(df_all_details)
    actors_df=tweak_cast(df_all_casting,df)
    director_df=tweak_director(df_all_casting,df)
    
    data_awards_cleaned=data_awards_cleaned_func(df_awards)
    df=(
    df
    .merge(actors_df,on=['id'],how="left")
    .merge(director_df,on=['id'],how="left")
    # .pipe(get_var, 'new_cols') 
    .drop(columns=[
        # "id",
        "genres","spoken_languages_number"] )
    ).assign(
        fame_nominated= lambda df_:df_.apply(lambda df_:fame_func(data_awards_cleaned,df_.director_name,df_.year),axis=1),
        fame_won= lambda df_:df_.apply(lambda df_:fame_func(data_awards_cleaned,df_.director_name,df_.year,stat=1),axis=1)
        
    ).drop(columns=["director_name","year"] )


    df_nulls=df.replace(-999,np.NAN)
    df
    
    return df,df_nulls

def tweak_data4_prediction(df_all_casting,record_casting,df_all_details,record_data,df_awards):
    record_data.iloc[0,0]=-1001
    record_data.reset_index(inplace=True,drop=True)
    record_casting.iloc[0,0]=-1001
    record_casting.reset_index(inplace=True,drop=True)
    df_all_casting=pd.concat([df_all_casting.copy(deep=True),record_casting])
    df_all_details=pd.concat([df_all_details.copy(deep=True),record_data])

    df=tweak_details(df_all_details)

    actors_df=tweak_cast(df_all_casting,df)

    director_df=tweak_director(df_all_casting,df)

    data_awards_cleaned=data_awards_cleaned_func(df_awards)
    df=(
    df
    .merge(actors_df,on=['id'],how="left")
    .merge(director_df,on=['id'],how="left")
    .drop(columns=[
        # "id",
        "genres","spoken_languages_number"] )
    ).assign(
        fame_nominated= lambda df_:df_.apply(lambda df_:fame_func(data_awards_cleaned,df_.director_name,df_.year),axis=1),
        fame_won= lambda df_:df_.apply(lambda df_:fame_func(data_awards_cleaned,df_.director_name,df_.year,stat=1),axis=1)
        
    ).drop(columns=["director_name","year"] )

    
    
    
    return df.query("id==-1001").drop(columns=["id","revenue","popularity"])