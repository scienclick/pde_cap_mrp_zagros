import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pickle
import warnings
import shap.plots
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn import set_config
from mrputils.loaders import DataLoader
from mrputils.processors import tweak_data,tweak_data4_prediction

#import sys

#====================================================================
#region imports
st.set_page_config(layout='centered', page_title="Zagros PDE", page_icon='🎬')
set_config(display="diagram")

RANDOMSEED = 100
DISPLAY_WIDTH = 400
DISPLAYMAX_COLUMNS = 25

random.seed(RANDOMSEED)
pd.set_option('display.width', DISPLAY_WIDTH)
pd.set_option('display.max_columns', DISPLAYMAX_COLUMNS)

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')

# %pwd # to see thecurrent directory
# %cd '/home/abdurraouf/code/Abdurraouf/pde_cap_mrp_zagros'

#endregion

popularity = 0
revenue = 0

#region == Import The Data ======================================================================================

if "dp" not in st.session_state:
    @st.cache()
    def get_dp():
        dp=DataLoader("data")
        dp.get_data()
        return dp
    st.session_state["dp"] = get_dp()

 #loading all the data

dp = st.session_state["dp"]
df_awards,df_all_casting,df_all_details,data_awards_cleaned=dp.df_awards,dp.df_all_casting,dp.df_all_details,dp.data_awards_cleaned


@st.cache()
def tweak_data_wrapper(df_all_casting,df_all_details,df_awards):
    return tweak_data(df_all_casting,df_all_details,df_awards)
__,df_nulls=tweak_data_wrapper(df_all_casting,df_all_details,data_awards_cleaned)
#endregion

#region == Load The Models ======================================================================================


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_models():
    M=pickle.load(open('models/finalized_model.sav',"rb")) # model for revenue
    preprocessor=pickle.load(open('models/preprocessor_x.sav',"rb")) #processor #1
    preprocessor_with_id=pickle.load(open('models/preprocessor_x_id.sav',"rb")) # processor #2
    M_pop=pickle.load(open('models/popularity.sav',"rb"))#model popularity
    knn_model=pickle.load(open('models/knn_similarity.sav',"rb"))
    m_fit=pickle.load(open('models/final_estimator.sav',"rb"))#for shap analysis

    return (M, preprocessor, preprocessor_with_id, M_pop, knn_model, m_fit)

models = get_models()

M = models[0]
preprocessor = models[1]
preprocessor_with_id = models[2]
M_pop = models[3]
knn_model = models[4]
m_fit = models[5]
#endregion

#=========================================================================================================

# Set layout , title, and icon.


# This line of code will read the css style from style.css file.
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



#=================================================
#                    Sidebar                     #
#=================================================

# region Sidebar

# SLB Logo
st.sidebar.image('https://seeklogo.com/images/S/slb-2022-logo-39A081F6E8-seeklogo.com.png',width=80)

st.sidebar.write("___")

@st.cache()
def estimate(budget_input, genres, actors, director):
    #== sample record format for prediction; sample data called xx ===========================================
    id=597
    record_data=df_all_details.query("id==@id")
    record_casting=df_all_casting.query("id==@id")
    record_data["genres"] = "|".join(genres)
    record_data["budget"] = budget_input
    record_casting["actor1_name"] = actors[0]
    record_casting["actor2_name"] = actors[1]
    record_casting["actor3_name"] = actors[2]
    record_casting["actor4_name"] = actors[3]
    record_casting["actor5_name"] = actors[4]
    record_casting["director_name"] = director
    xx=tweak_data4_prediction(df_all_casting,record_casting,df_all_details,record_data,data_awards_cleaned)

    X_with_id=df_nulls.drop(columns=["revenue","popularity"])
    X_with_id_processed=pd.DataFrame(preprocessor_with_id.transform(X_with_id),columns=preprocessor_with_id.get_feature_names_out())


    X=df_nulls.drop(columns=["id","revenue","popularity"])
    X_processed=pd.DataFrame(preprocessor.transform(X),columns=preprocessor.get_feature_names_out())

    st.session_state['X_processed'] = X_processed
    st.session_state['X_with_id_processed'] = X_with_id_processed


    # Find Similar movies

    num_neighbors=5
    xx_processed=preprocessor.transform(xx)#sample processed
    similars=list(knn_model.kneighbors(xx_processed,n_neighbors=num_neighbors)[1][0])
    similar_ids=list(X_with_id_processed.iloc[similars].remainder__id.values)
    similar_movies = df_all_details.query('id  in @similar_ids')[["id","title","genres","release_date","revenue","popularity"]].merge(
        df_all_casting.query('id  in @similar_ids')[["id","director_name","actor1_name","actor2_name","actor3_name"]],on="id"
    )

    st.session_state['similar_movies']=similar_movies

    popularity = M_pop.predict(xx)
    revenue = M.predict(xx)[0]

    return (revenue,popularity,similar_ids, similar_movies)



# Input from User


#-------------------------------------------

# Budget
budget_input = st.sidebar.slider(label='Budget in Million (USD):',value=20, max_value=200, step=5, on_change=None)
budget_input=budget_input*1000_000

st.sidebar.write("___")

# [Directors List]
directors_list=sorted(['James Cameron','Steven Spielberg','Don Hertzfeldt','Clint Eastwood','Martin Scorsese',
                'Woody Allen','Joel Coen','Quentin Tarantino','Pedro Almodóvar','Peter Jackson','Christopher Nolan',
                "James Bobin", "Roar Uthaug", "F. Gary Gray"])
director = st.sidebar.selectbox('Director:', directors_list)

# [Actors List]
actors_list = sorted(['Kathy Bates','Billy Zane','Frances Fisher','Leonardo DiCaprio','Kate Winslet','Mel Blanc','Sivaji Ganesan',
                      'James A. FitzPatrick','Oliver Hardy','Mammootty','Charles Starrett','M. G. Ramachandran','Gemini Ganesan',
                      'Johnny Mack Brown','Pinto Colvig',"Isabela Merced", "Jeffrey Wahlberg", "Madeleine Madden", "Eugenio Derbez", "Michael Pena",
                      "Alicia Vikander", "Dominic West", "Walton Goggins",
               "Daniel Wu", "Kristin Scott Thomas", "Chris Hemsworth",

               "Tessa Thompson", "Rebecca Ferguson", "Kumail Nanjiani", "Rafe Spall"])
actors = st.sidebar.multiselect('Actors',actors_list,max_selections=5,default=actors_list[:5])


# [Genres]
genres_list= sorted(['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy',
              'foreign', 'history', 'horror', 'movie', 'music', 'mystery', 'romance', 'science', 'thriller', 'tv', 'war', 'western'])
genres = st.sidebar.multiselect('Genres',genres_list, default=genres_list[0])

st.sidebar.write("___")

#endregion

#================================================
#                 Main Page                     #
#================================================

# Header (Project Name and Description)
headcol1, headcol2 = st.columns([5,1])

with headcol1:
    # Title
    st.title('Movie Revenue Prediction')

    # Description
    st.write("""
    ### Description:
    This project will help you predict the **revenue** for a movie based on some parameters like; director, actors, month of the year ... etc.
    """)

with headcol2:
    # Team Logo
    st.write("")
    # st.image("https://i.ibb.co/mchjR4k/zagros-icon.png")


# ==============================================

#region ==== ESTIMATE =====
button = st.sidebar.button('Estimate')

if button:
    if 'similar_movies' in st.session_state:
        del st.session_state['Similar_Movies']
    estimation= estimate(budget_input,genres,actors, director)
    revenue = estimation[0]
    popularity = estimation[1]
    similar_ids = estimation[2]
    similar_movies = estimation[3]

    st.session_state['Similar_Movies'] = similar_movies

    #==== Hist Plot ==============================

    #make this example reproducible
    np.random.seed(0)

    #create data
    x = np.random.normal(loc = 61737022.44220184, scale = 139664301.60821903, size =1000)

    #create normal distribution curve
    fig1, ax = plt.subplots(figsize=(5,4))
    fig1 = sns.displot(x, kde=True)
    plt.axvline(revenue, color='red', linestyle='dashed', linewidth=2, label='valueeeee')
    ymin, ymax = plt.ylim()
    plt.text(revenue,ymax*0.7,f'${round(revenue/1000_000,2)}MM',rotation=90)
    plt.ylabel("Count")
    plt.xlabel("Revenue in $MM")

    st.pyplot(fig1)

    tcol1, tcol2, tcol3 = st.columns([1,1,1])

    with tcol1:
        st.write(f"""
        ### Revenue:
        ### ${'{:,}'.format(round(revenue,2))}
        """)

    profit = revenue - budget_input
    with tcol2:
        st.write(f"""
        ### Profit($):
        ### ${'{:,}'.format(profit)}
        """)

    with tcol3:
        st.write(f"""
        ### Popularity 🎭:
        ### {str(round(popularity[0],2))} /10
        """)

    gauge_max=10
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = round(popularity[0],2),
    mode = "gauge+number",
    title = {'text': "Popularity"},
    delta = {'reference': 1},
    gauge = {'axis': {'range': [None, gauge_max]},
             'steps' : [
                 {'range': [0, 3], 'color': "red"},
                 {'range': [3, 5], 'color': "orange"},
                 {'range': [5, 7], 'color': "yellow"},
                 {'range': [7, 10], 'color': "green"}],
             'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': round(popularity[0],2)},
             'bar': {'color': "white", 'thickness':0.25}}))

    st.plotly_chart(fig)

    #endregion

    # st.write(similar_ids)
    # st.write(similar_movies)

# similar_movies = st.session_state['similar_movies']

st.write('---')
st.subheader('Similar Movies')
similar_movies = st.session_state['Similar_Movies']
counter = 0
for movie in st.session_state['Similar_Movies']['title']:
    counter+=1
    st.write(f'{counter}) {movie}')

with st.expander("Click the drop-down and select a movie for SHAP Analysis.",expanded=False):
    selected_movie = st.selectbox('Select a movie:',similar_movies['title'])

X_processed = st.session_state['X_processed']
X_with_id_processed = st.session_state['X_with_id_processed']

shap_values = np.load('models/shap_values.npy') # load
expected_value = float(np.load('models/expected_value.npy')) # load

# # index of the selected movie
id = list(similar_movies['title']).index(selected_movie)
i = st.session_state['Similar_Movies'].iloc[id]['id']


# Debugging only
print(st.session_state['Similar_Movies'])
print(id)
print(i)
# _________________________________________________________

ii=X_with_id_processed[X_with_id_processed.remainder__id==i ].index.values[0]
ss=pd.Series(shap_values[ii],index=X_processed.columns)
s1=np.sign(ss)
s2=ss.map(lambda x : x).abs().sort_values(ascending = False)
s3=s2*s1
s3=s3.reindex(s2.index)
S=s3[0:13]
S=S.append(pd.Series([s3[13:].sum()],index=["__Rest_of_features"]))
S=S.rename(index={k: k.lower().split('__')[1] for k in S.index})

import plotly.graph_objects as go

fig = go.Figure(go.Waterfall(
    name = "2018", orientation = "h", measure = ["relative"]*len(S),
    y = S.index,
    x = S.values,
    connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
))

fig.update_layout(title = "Profit and loss statement 2018")

st.write(fig)
