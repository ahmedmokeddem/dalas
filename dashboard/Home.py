import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide"
)


st.markdown("<h1 style='text-align: center; color: white;'>L'intelligence est elle acquise ou innée ?</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 0.5])

st.sidebar.success("Select a demo above.")
df=pd.read_csv("../datasets_merged/df_mergedFinal.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)
with col1:
    pass

with col2:
    df_filtered = df.loc[(~df['Averageiqbycountry_Iqlynnbecker2019'].isna())&(~df['Continent'].isna())]
    df_filtered=df_filtered[['Country','Continent','Averageiqbycountry_Iqlynnbecker2019']]
    fig = px.histogram(df_filtered, x='Averageiqbycountry_Iqlynnbecker2019', color='Continent', nbins=30, 
                   histnorm='probability density', 
                   title='Approximation de la courbe de densité du quotient intellectuel (QI)')

    # Renommer l'axe des x
    fig.update_xaxes(title_text='Quotient intellectuel (QI)')
    st.plotly_chart(fig, use_container_width=True)
    