
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Clustering",
    page_icon="🧩", 
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: white;'>Regroupement des pays en clusters</h1>", unsafe_allow_html=True)
df=pd.read_csv("../datasets_merged/df_mergedFinal.csv")
numerical_columns = df.select_dtypes(include=['number'])
df_num=numerical_columns.drop(columns=['Annee_Souverainete','Break_1','Break_2', 'Break_3', 'Break_4', 'Break_5','Political_Regime', 'Mean_Rank_Univ'])
df_num.drop(['Averageiq_Ici2017Score', 'Averageiqpisa2022Meanscoremathematics','Averageiqpisa2022Meanscorereading','Averageiqpisa2022Meanscorescience'], axis=1, inplace=True)

# Supprimer les colonnes avec moind de 130 valeurs rensignées
df_num.drop(['Primary',"Lower_Secondary","End_Of_The_School_Year_Break","Literacy_Rate_2021","Global_rank_Literacy_rate","Nb_graduates","Nb_Foreign_Students" ], axis=1, inplace=True)

# Garder une trace des noms de pays et continents
df_num_country_continent=df_num.copy()
df_num_country_continent['Country']=df.Country
df_num_country_continent['Continent']=df.Continent

df_num_clean=df_num.dropna()
df_num_country_continent=df_num_country_continent.dropna()

# Normaliser les données
scaler = StandardScaler()
df_num_norm=pd.DataFrame(scaler.fit_transform(df_num_clean), columns=df_num_clean.columns)

kmeans = KMeans(n_clusters = 5, random_state=10)
kmeans.fit(df_num_norm)

# Afficher le Silhouette score
#silhouette_avg = silhouette_score(df_num_norm, kmeans.labels_)
#print("Le score de silhouette moyen est :", silhouette_avg)

# Afficher les statistiques par rapport à chaque cluster (note : effectuer l'analyse sur le dataset avant normalisation)
clusters_desc = df_num_clean.assign(classe = kmeans.labels_) 
clusters_desc.groupby("classe").mean()


# Afficher pour chaque classe, les pays qui y appartiennent
clusters_desc["Country"]=df_num_country_continent['Country']
clusters_desc["Continent"]=df_num_country_continent['Continent']
df_kmeans=clusters_desc.loc[:,["Country", 'Continent','classe']]

# Charger les données géographiques intégrées de Plotly
df_plotly = px.data.gapminder().query("year == 2007")
df_plotly.rename(columns={"country":"Country", 'continent':'Continent'}, inplace=True)

df_map=pd.merge(df_kmeans,df_plotly, on='Country', how='left')
df_map = df_map.sort_values(by='classe', ascending=False)
df_map.classe=df_map.classe.astype('string')

df_map.loc[df_map["Country"]=="DR Congo", "iso_alpha"]="COD"
df_map.loc[df_map["Country"]=="Ivory Coast", "iso_alpha"]="CIV"
df_map.loc[df_map["Country"]=="Republic of the Congo", "iso_alpha"]="COG"
df_map.loc[df_map["Country"]=="Slovakia", "iso_alpha"]="SVK"


fig = px.choropleth(df_map,
                    locations="iso_alpha",
                    hover_name="Country",
                    color="classe",
                    hover_data=["lifeExp", "gdpPercap"])

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    width=1000,
    height=800
)
fig.update_geos(projection_type="natural earth", bgcolor='rgba(0,0,0,0)')

st.plotly_chart(fig, use_container_width=True)










