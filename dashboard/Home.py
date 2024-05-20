import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Specify the directory you want to list
directory = 'path/to/your/directory'

# List the contents of the directory
try:
    with os.scandir(directory) as entries:
        for entry in entries:
            print(entry.name)
except FileNotFoundError:
    print(f"The directory {directory} does not exist")
except PermissionError:
    print(f"You do not have permissions to access {directory}")
    
    
st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)


df=pd.read_csv("../datasets_merged/df_mergedFinal.csv")
df.loc[df["Country"]=="Antigua and Barbuda", 'Continent']="north-america"
df.loc[df["Country"]=="British Virgin Islands", 'Continent']="north-america"

df.loc[df["Country"]=="Cayman Islands", 'Continent']="north-america"
df.loc[df["Country"]=="Cook Islands", 'Continent']="oceania"

df.loc[df["Country"]=="Eswatini", 'Continent']="africa"
df.loc[df["Country"]=="Greenland", 'Continent']="north-america"

df.loc[df["Country"]=="Macau", 'Continent']="asia"
df.loc[df["Country"]=="New Caledonia", 'Continent']="oceania"

df.loc[df["Country"]=="Northern Mariana Islands", 'Continent']="oceania"
df.loc[df["Country"]=="Saint Kitts and Nevis", 'Continent']="north-america"

df.loc[df["Country"]=="Saint Lucia", 'Continent']="north-america"
df.loc[df["Country"]=="Saint Vincent and the Grenadines", 'Continent']="north-america"

df.loc[df["Country"]=="Sao Tome and Principe", 'Continent']="africa"
df.loc[df["Country"]=="Turks and Caicos Islands", 'Continent']="north-america"
df.loc[df["Country"]=="British Virgin Islands", 'Gdp']=1.027*10**9
df.loc[df["Country"]=="Cook Islands", 'Gdp']=287*10**6
df.loc[df["Country"]=="Kyrgyzstan", 'Gdp']= 9.371*10**6

df.loc[df["Country"]=="North Korea", 'Gdp']=18*10**9
df.loc[df["Country"]=="Palestine", 'Gdp']=17.13*10**9
df.loc[df["Country"]=="Taiwan", 'Gdp']=611.396*10**6
df_continent_gdp=df.loc[:, ["Continent", "Gdp"]].groupby("Continent").mean().reset_index().sort_values(by="Gdp")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h1 style='text-align: center; color: white;'>L'intelligence est elle acquise ou innée ?</h1>", unsafe_allow_html=True)
df_filtered = df.loc[(~df['Averageiqbycountry_Iqlynnbecker2019'].isna())&(~df['Continent'].isna())]
df_filtered=df_filtered[['Country','Continent','Averageiqbycountry_Iqlynnbecker2019']]
fig = px.histogram(df_filtered, x='Averageiqbycountry_Iqlynnbecker2019', color='Continent', nbins=30, 
                histnorm='probability density', 
                title='Approximation de la courbe de densité du quotient intellectuel (QI)')

# Renommer l'axe des x
fig.update_xaxes(title_text='Quotient intellectuel (QI)')
st.plotly_chart(fig, use_container_width=True)
st.markdown("<h2 style='text-align: center; color: white;'>Axe géoeconomique</h2>", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        fig_continent_gdp = px.pie(df_continent_gdp, values='Gdp', names='Continent', title='PIB moyen par continent')
        st.plotly_chart(fig_continent_gdp,use_container_width=True)

    with col2:
        df_country_gdp=df[df.Gdp>df_continent_gdp.Gdp.mean()*2]
        fig_country_gdp = px.pie(df_country_gdp, values='Gdp', names='Country', title='PIB moyen par pays')
        st.plotly_chart(fig_country_gdp,use_container_width=True)

st.markdown("<h2 style='text-align: center; color: white;'>Axe historique</h2>", unsafe_allow_html=True)

with st.container(border=True):
    df_gdp=pd.read_csv("../datasets_dw/Ahmed/dataset GDP.csv", skiprows=[0,1,2])
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
    df=pd.read_csv("../datasets_merged/df_mergedFinal.csv")
    df_contient_country=df.loc[:,['Country', 'Continent']]
    df_gdp_continent_interm=pd.merge(df_contient_country, df_gdp, on='Country' )
    df_gdp_continent_interm=df_gdp_continent_interm.drop(['Country'],axis=1)
    df_gdp_continent=df_gdp_continent_interm.groupby('Continent').mean()
    df_gdp_continentT=df_gdp_continent.transpose()
    df_gdp_continentT=df_gdp_continentT.reset_index()
    fig = px.line(df_gdp_continentT, x="index", y="africa")
    for continent in df_gdp_continentT.columns[1:]:
        fig.add_scatter(x=df_gdp_continentT["index"], y=df_gdp_continentT[continent], mode='lines', name=continent)
    fig.update_layout(xaxis_title='Continents', yaxis_title='PIB moyen', title='Evolution du PIB moyen par continent')
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        df_filtered = df.loc[(~df['Indice_Evolution'].isna()) & (~df['Annee_Souverainete'].isna())]
        df_filtered=df_filtered[['Country','Continent','Indice_Evolution', 'Annee_Souverainete']]
        df_sorted = df_filtered.sort_values(by='Annee_Souverainete')

        # Tracer le scatter plot
        fig = px.scatter(df_sorted, x='Annee_Souverainete', y='Indice_Evolution',
                        title='Indice d\'évolution en fonction de l\'année de souveraineté',
                        labels={'Annee_Souverainete': 'Année de souveraineté', 'Indice_Evolution': 'Indice d\'évolution'},
                        color=df_sorted['Continent'])
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        df_continent_indiceEvol=df.loc[:, ["Continent", "Indice_Evolution"]].groupby("Continent").mean().reset_index().sort_values(by="Indice_Evolution")
        fig_continent_indiceEvol = px.bar(df_continent_indiceEvol, x='Continent', y='Indice_Evolution',  title='Indice Evolution moyen par continent')
        st.plotly_chart(fig_continent_indiceEvol,use_container_width=True)

st.markdown("<h2 style='text-align: center; color: white;'>Axe socio-économique</h2>", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        df['Budget_Education_Nationale'] = df['Gdp'] * df['Education_Spending_2021']/ 100
        df_filtered = df.loc[(~df['Gdp'].isna()) & (~df['Budget_Education_Nationale'].isna())]
        fig = px.scatter(df_filtered, x='Gdp', y='Budget_Education_Nationale', text='Country', title='Relation entre le PIB et le budget éducatif')
        fig.update_traces(textposition='top center')  # Position du texte au-dessus de chaque point
        fig.update_layout(xaxis_title="PIB (en billions de dollars)", yaxis_title="Budget éducation nationale")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_filtered = df.loc[(~df['Political_Regime'].isna()) & (~df['Education_Spending_2021'].isna())]
        df_filtered=df_filtered[['Political_Regime','Education_Spending_2021']]
        regime_mapping = {
            0: 'Closed Autocracy',
            1: 'Electoral Autocracy',
            2: 'Electoral Democracy',
            3: 'Liberal Democracy'
        }

        # Ajouter une nouvelle colonne 'Regime_Label' contenant les libellés correspondants
        df_filtered['Regime_Label'] = df_filtered['Political_Regime'].map(regime_mapping)

        # Tracer le boxplot
        fig = px.box(df_filtered, x='Regime_Label', y='Education_Spending_2021', 
                    title='Distribution du pourcentage du budget de l\'éducation nationale par régime politique',
                    labels={'Regime_Label': 'Régime politique', 'Education_Spending_2021': 'Budget éducatif (% du PIB)'})

        fig.update_traces(marker=dict(color='rgb(158,202,225)', outliercolor='rgba(219, 64, 82, 0.6)', line=dict(outliercolor='rgba(219, 64, 82, 0.6)', outlierwidth=2)))
        st.plotly_chart(fig, use_container_width=True)
    df_filtered = df.loc[(~df['Gdp'].isna()) & (~df['Immigrationbycountry_Immigrants'].isna())& (~df['Averageiqbycountry_Iqlynnbecker2019'].isna())& (~df['Immigrationbycountry_Emigrants'].isna())]
    df_filtered=df_filtered[['Country','Gdp', 'Immigrationbycountry_Immigrants', 'Immigrationbycountry_Emigrants', 'Averageiqbycountry_Iqlynnbecker2019']]
    fig = px.scatter_matrix(df_filtered, dimensions=['Gdp', 'Immigrationbycountry_Immigrants', 'Immigrationbycountry_Emigrants', 'Averageiqbycountry_Iqlynnbecker2019'],
                            labels={'Gdp': 'GDP', 'Immigrationbycountry_Immigrants': 'Immigrants In', 'Immigrationbycountry_Emigrants': 'Immigrants Out', 'Averageiqbycountry_Iqlynnbecker2019': 'Average IQ'},
                            title='Scatter Plot Matrix pour les variables analysées',
                            color=df_filtered['Country'],height=800)
                        
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<h2 style='text-align: center; color: white;'>Axe culturel et educatif</h2>", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        df_filtered = df.loc[(~df['Literacy_Rate_2021'].isna()) & (~df['Education_Spending_2021'].isna())]
        df_filtered=df_filtered[['Country','Continent','Literacy_Rate_2021', 'Education_Spending_2021']]
        df_filtered['Ratio_LiteracyRate_EducationSpending']= df_filtered['Literacy_Rate_2021']/df_filtered['Education_Spending_2021']
        fig = px.bar(df_filtered, x='Country', y=['Ratio_LiteracyRate_EducationSpending'],
             title='Ratio d\'alphabétisation sur pourcentage du budget éducatif par pays',
             labels={'value': 'Valeur', 'variable': 'Catégorie'},
             barmode='group')

        # Tri par la variable 'Literacy_Rate_2021'
        fig.update_xaxes(categoryorder='max descending')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_interm=df.loc[:, ["Country","Continent", "Literacy_Rate_2021", "Nbvotes_Musees_Clean", "Notes_Musees"]]
        df["Nbvotes_Musees_Clean"]=df_interm["Nbvotes_Musees_Clean"]
        df["Notes_Musees"]=df_interm["Notes_Musees"]
        df_interm['Indice_Culturel']=df_interm["Nbvotes_Musees_Clean"]*df_interm["Notes_Musees"]
        df_interm=df_interm.loc[:, ["Country","Continent", "Literacy_Rate_2021", "Indice_Culturel"]]
        fig_indiceCulturel_TauxAlphab = px.scatter(df_interm, x="Literacy_Rate_2021", y="Indice_Culturel", color="Continent", symbol="Country", title="Indice culturel et taux d'alphabétisme",width=500)
        st.plotly_chart(fig_indiceCulturel_TauxAlphab, use_container_width=True)



    df['Pourcentage_immigres'] = (df['Immigrationbycountry_Immigrants'] / df['Population']) * 100
    df_filtered = df.loc[(~df['Pourcentage_immigres'].isna()) & (~df['Nb_Foreign_Students'].isna()) ]
    df_filtered=df_filtered[['Country','Pourcentage_immigres', 'Nb_Foreign_Students']]
    fig = px.bar(df_filtered, x='Country', y=['Pourcentage_immigres', 'Nb_Foreign_Students'],
             title='Proportion des étudiants étrangers par rapport aux immigrants par pays',
             labels={'value': 'Nombre', 'variable': 'Catégorie'},
             barmode='group')
    fig.update_xaxes(categoryorder='max descending')
    st.plotly_chart(fig, use_container_width=True)




