from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pickle
import warnings
import shap
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn import set_config
from mrputils.loaders import DataLoader
from mrputils.processors import tweak_data,tweak_data4_prediction
import sys
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
#== Import The Data ======================================================================================

if "dp" not in st.session_state:
    # @st.cache
    def get_dp():
        dp=DataLoader("data")
        dp.get_data()
        return dp
    st.session_state["dp"] = get_dp()

 #loading all the data

dp = st.session_state["dp"]
df_awards,df_all_casting,df_all_details=dp.df_awards,dp.df_all_casting,dp.df_all_details

@st.cache
def tweak_data_wrapper(df_all_casting,df_all_details,df_awards):
    return tweak_data(df_all_casting,df_all_details,df_awards)
__,df_nulls=tweak_data_wrapper(df_all_casting,df_all_details,df_awards)


#== Load The Models ======================================================================================

# Use st.session_state and @st.cache


def get_models(): 
    M=pickle.load(open('models/finalized_model.sav',"rb")) # model for revenue
    preprocessor=pickle.load(open('models/preprocessor_x.sav',"rb")) #processor #1
    preprocessor_with_id=pickle.load(open('models/preprocessor_x_id.sav',"rb")) # processor #2
    M_pop=pickle.load(open('models/popularity.sav',"rb"))#model popularity
    knn_model=pickle.load(open('models/knn_similarity.sav',"rb"))
    m_fit=pickle.load(open('models/final_estimator.sav',"rb"))#for shap analysis
    
    return (M, preprocessor, preprocessor_with_id, M_pop, knn_model, m_fit)

M = get_models()[0]
preprocessor = get_models()[1]
preprocessor_with_id = get_models()[2]
M_pop = get_models()[3]
knn_model = get_models()[4]
m_fit = get_models()[5]
#=========================================================================================================

# Set layout , title, and icon.


# This line of code will read the css style from style.css file.
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# SLB Logo
st.sidebar.image('https://seeklogo.com/images/S/slb-2022-logo-39A081F6E8-seeklogo.com.png',width=80)

st.sidebar.write("___")

#=================================================
#                    Sidebar                     #
#=================================================

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
    xx=tweak_data4_prediction(df_all_casting,record_casting,df_all_details,record_data,df_awards)
    popularity = M_pop.predict(xx)
    revenue = M.predict(xx)[0]
    return (revenue,popularity)

  
# Input from User
#-------------------------------------------

# Budget Text_Input
budget_input = st.sidebar.slider(label='Budget in Million (USD):',value=20, max_value=200, step=5, on_change=None)
budget_input=budget_input*1000_000

st.sidebar.write("___")

# Directors List
directors_list=sorted(['James Cameron','Steven Spielberg','Don Hertzfeldt','Clint Eastwood','Martin Scorsese',
                'Woody Allen','Joel Coen','Quentin Tarantino','Pedro Almodóvar','Peter Jackson','Christopher Nolan',
                "James Bobin", "Roar Uthaug", "F. Gary Gray"])
director = st.sidebar.selectbox('Director:', directors_list)

# Actors List
actors_list = sorted(['Kathy Bates','Billy Zane','Frances Fisher','Leonardo DiCaprio','Kate Winslet','Mel Blanc','Sivaji Ganesan',
                      'James A. FitzPatrick','Oliver Hardy','Mammootty','Charles Starrett','M. G. Ramachandran','Gemini Ganesan',
                      'Johnny Mack Brown','Pinto Colvig',"Isabela Merced", "Jeffrey Wahlberg", "Madeleine Madden", "Eugenio Derbez", "Michael Pena",
                      "Alicia Vikander", "Dominic West", "Walton Goggins",

               "Daniel Wu", "Kristin Scott Thomas", "Chris Hemsworth",

               "Tessa Thompson", "Rebecca Ferguson", "Kumail Nanjiani", "Rafe Spall"])
actors = st.sidebar.multiselect('Actors',actors_list,max_selections=5,default=actors_list[:5])

# Genres
genres_list= sorted(['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy', 
              'foreign', 'history', 'horror', 'movie', 'music', 'mystery', 'romance', 'science', 'thriller', 'tv', 'war', 'western'])
genres = st.sidebar.multiselect('Genres',genres_list, default=genres_list[0])

st.sidebar.write("___")


#================================================
#                 Main Page                     #
#================================================

# Header (Project Name and Description)
headcol1, headcol2 = st.columns([5,1])

with headcol1:
    # Title
    st.title('Movie Revenue Project')
    
    # Description
    st.write("""
    ### Description:
    This project will help you predict the **revenue** for a movie based on some parameters like; director, actors, month of the year ... etc.
    """)
    
with headcol2:
    # Team Logo
    st.image("https://i.ibb.co/mchjR4k/zagros-icon.png")
    

# ==============================================

if st.sidebar.button('Estimate'):
    revenue = estimate(budget_input,genres,actors, director)[0]
    popularity = estimate(budget_input,genres,actors, director)[1]
    
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
    plt.ylabel("")

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
    