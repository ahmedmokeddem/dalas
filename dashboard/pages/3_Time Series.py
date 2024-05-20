# python -m pip install prophet
# conda install -c conda-forge prophet
# python -m pip install "prophet==1.1.2"


import numpy as np             
import pandas as pd            
import matplotlib.pylab as plt           
from prophet import Prophet
import streamlit as st
import holidays



st.set_page_config(
    page_title="Séries Temporelles",
    page_icon="📈", 
    layout="wide"
)


st.markdown("<h1 style='text-align: center; color: white;'>Analyse temporelle de de l'indice d'évolution des pays</h1>", unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)



# Charger les données
df_gdp=pd.read_csv("datasets_dw/Ahmed/dataset GDP.csv", skiprows=[0,1,2])
df_gdp=df_gdp.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1).iloc[:,:-1]
df_gdp=df_gdp.rename(columns={'Country Name':'Country'})
df_gdp.loc[df_gdp['Country'] == 'Congo, Dem. Rep.', 'Country'] = "DR Congo"
df_gdp.loc[df_gdp['Country'] == 'Viet Nam', 'Country'] = "Vietnam"
df_gdp.loc[df_gdp['Country'] == 'Turkiye', 'Country'] = "Turkey"
df_gdp.loc[df_gdp['Country'] == 'Korea, Rep.', 'Country'] = "South Korea"
df_gdp.loc[df_gdp['Country'] == "Cote d'Ivoire", 'Country'] = "Ivory Coast"
df_gdp.loc[df_gdp['Country'] == 'Czechia', 'Country'] = "Czech Republic"
df_gdp.loc[df_gdp['Country']=="Lao PDR","Country"]="Laos"
df_gdp.loc[df_gdp['Country']=="Congo, Rep.","Country"]="Republic of the Congo"
df_gdp.loc[df_gdp['Country'] == 'Slovak Republic', 'Country'] = "Slovakia"
df_gdp.loc[df_gdp['Country'] == 'Macao SAR, China', 'Country'] = "Macau"
df_gdp.loc[df_gdp['Country'] == 'Cabo Verde', 'Country'] = "Cape Verde"
df_gdp.loc[df_gdp['Country'] == 'St. Lucia', 'Country'] = "Saint Lucia"
df_gdp.loc[df_gdp['Country'] == 'St. Vincent and the Grenadines', 'Country'] = "Saint Vincent and the Grenadines"
df_gdp.loc[df_gdp['Country'] == 'St. Kitts and Nevis', 'Country'] = "Saint Kitts and Nevis"

df=pd.read_csv("datasets_merged/df_mergedFinal.csv")


df_contient_country=df.loc[:,['Country', 'Continent']]
df_gdp=pd.merge(df_contient_country, df_gdp, on='Country' )

df_gdpT=df_gdp.transpose()
df_gdpT=df_gdpT.reset_index()
df_gdpT.columns=["ds"]+list(df_gdpT.iloc[0,1:])
continents=df_gdpT.iloc[1,1:]

# Supprimer les lignes des pays et continents pour ne garder que la date et les GDP
df_gdpT=df_gdpT.iloc[2:,:]



col1, col2 = st.columns([0.5, 0.5])

# Options de la colonne spécifique dans le DataFrame
options = df_gdp["Country"].unique()  # Supposons que votre colonne s'appelle "Options"

with col1:
    question1 = "Choisissez le 1e pays"  # Définir votre question ici
    # Liste déroulante pour la sélection de l'option
    choix1 = st.selectbox(question1, options, index=list(options).index('France'))

with col2:
    question2 = "Choisissez le 2d pays"  # Définir votre question ici
    # Liste déroulante pour la sélection de l'option
    choix2 = st.selectbox(question2, options, index=list(options).index('Singapore'))

    




st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)

def end_year_input():
    end_year = st.number_input("Prédire jusqu'à l'année ...", min_value=1960, max_value=2090, value=2036)
    return end_year

end_year = end_year_input()



pays=choix1
annee_souverainete=list(df[df.loc[:,'Country']==pays]['Annee_Souverainete'])[0]
df_firstCountry=df_gdpT.loc[:,['ds', pays]]
df_firstCountry.rename(columns={pays:'y'}, inplace=True)
df_firstCountry.y=df_firstCountry.y/annee_souverainete

pays=choix2
annee_souverainete=list(df[df.loc[:,'Country']==pays]['Annee_Souverainete'])[0]
df_secondCountry=df_gdpT.loc[:,['ds', pays]]
df_secondCountry.rename(columns={pays:'y'}, inplace=True)
df_secondCountry.y=df_secondCountry.y/annee_souverainete

# Remplacer les NaN par la dernière valeur enregistrée
df_firstCountry['y'].fillna(method='ffill', inplace=True)
df_secondCountry['y'].fillna(method='ffill', inplace=True)

# Appliquer le logarithme
df_firstCountry['y'] = np.log(df_firstCountry['y']).astype('float64')  # Convertir en float64 pour gérer les NaN et inf
df_secondCountry['y'] = np.log(df_secondCountry['y']).astype('float64')

# Remplacer les valeurs non-finites par NaN
df_firstCountry['y'].replace([np.inf, -np.inf], np.nan, inplace=True)
df_secondCountry['y'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Gérer les cas d'erreurs
if(len(df_firstCountry.y) - np.sum(df_firstCountry.y.isna()) < 5):
    st.error("Veuillez sélectionner une autre 1e pays car celui-ci ne dispose pas de suffisamment de données pour l'analyse")
    st.stop()
    
if(len(df_secondCountry.y) - np.sum(df_secondCountry.y.isna()) < 5):
    st.error("Veuillez sélectionner une autre 2d pays car celui-ci ne dispose pas de suffisamment de données pour l'analyse")
    st.stop()

# Instancier le modèle Prophet
prophet_model_1Country = Prophet()
prophet_model_1Country.fit(df_firstCountry)

prophet_model_2Country = Prophet()
prophet_model_2Country.fit(df_secondCountry)

nb_year_future = np.max([int(end_year) - int(df_firstCountry.ds.values.max()), int(end_year) - int(df_secondCountry.ds.values.max())])+1

if(nb_year_future<0):
    st.error("L'année de fin doit être supérieure à l'année actuelle")
    st.stop()
# Construire le dataset correspondant aux 10 prochaines années
future = prophet_model_1Country.make_future_dataframe(periods=365*nb_year_future)
future=future[future["ds"].dt.day==1]
future=future[future['ds'].dt.month==1 ]
future.tail()

# Prédire l'indice évolution sur les 10 prochaines années
forecast_firstCountry = prophet_model_1Country.predict(future)
forecast_firstCountry[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Prédire l'indice évolution sur les 10 prochaines années
forecast_secondCountry = prophet_model_2Country.predict(future)
forecast_secondCountry[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

# Set background color to transparent
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')

prophet_model_2Country.plot(forecast_secondCountry, ax=plt.gca()) 

prophet_model_1Country.plot(forecast_firstCountry, ax=plt.gca()) 
plt.gca().get_lines()[1].set_color('red')  

plt.legend([" ",choix1, " ",choix2])
ax.set_title("Indice d'évolution de " + choix1 + " et de " + choix2 + " dans le temps", color='white')
st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
st.pyplot(plt.gcf())
plt.close()

#plt.style.use('dark_background')
plt.title("Décomposition de l'indice d'évolution de " + choix1, color='white')
prophet_model_1Country.plot_components(forecast_firstCountry)
st.pyplot(plt.gcf())

plt.title("Décomposition de l'indice d'évolution de " + choix2, color='white')
prophet_model_2Country.plot_components(forecast_secondCountry)
st.pyplot(plt.gcf())













