import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self,path) -> None:
        print("starting class")
        self.path=path
        self.df_awards=None
        self.df_500_fav=None
        self.df_900_acc=None
        self.df_all_casting=None
        self.df_all_details=None
        self.df_lan2country_df=None
        self.df_most_common_lang=None
        self.df_spliberg=None

    def get_data(self):
        self.df_awards = pd.read_csv(self.path+"/220k_awards_by_directors.csv")
        self.df_500_fav = pd.read_csv(self.path+"/500 favorite directors_with wikipedia summary.csv", sep=";", header=0,
                                 names=['A', 'B'])
        self.df_900_acc = pd.read_csv(self.path+"/900_acclaimed_directors_awards.csv", sep=";")
        self.df_all_casting = pd.read_csv(self.path+"/AllMoviesCastingRaw.csv", sep=";")
        self.df_all_details = pd.read_csv(self.path+"/AllMoviesDetailsCleaned.csv", sep=";", low_memory=False)
        self.df_lan2country_df = pd.read_csv(self.path+'/language to country.csv')
        self.df_most_common_lang = pd.read_csv(self.path+"/MostCommonLanguageByDirector.csv")
        self.df_spliberg = pd.read_csv(self.path+"/spielberg_awards.csv", encoding='latin1')


    def ping(self):
        print(__file__)
        print(os.path.dirname(__file__))
        print(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data"))


if __name__ == "__main__":
    dp=DataLoader("data")
    dp.get_data()

    print()
