import streamlit as st
import pandas as pd
import numpy as np
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
pd.DataFrame.iteritems = pd.DataFrame.items
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import lime
import torch
import lime.lime_tabular
st.set_page_config(page_title="Regression", page_icon="📈",layout="wide")

class Regressor(nn.Module):
    def __init__(self, input_size=12,nb_neurones_par_couche=5,nb_couches=1):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(nb_couches):
            layers.append(nn.Linear(input_size, nb_neurones_par_couche))
            layers.append(nn.ReLU(inplace=True))
            input_size = nb_neurones_par_couche
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


st.markdown("<h1 style='text-align: center; '>Prédiction du QI</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([0.7, 0.3])



data_dashboard=pd.read_csv('dashboard/data/Dashboard_data_regression.csv')
data_dashboard_y=pd.read_csv('dashboard/data/Dashboard_data_regression_y.csv')
# Load the best model's state dictionary
best_model_path = 'dashboard/data/best_model.pth'
model = Regressor()
model.load_state_dict(torch.load(best_model_path))

# Ensure the model is in evaluation mode
model.eval()

# Define the predict_wrapper function
def predict_wrapper(x):
    x = torch.tensor(x, dtype=torch.float32)
    return model(x).detach().numpy()
feature_names=list(data_dashboard.iloc[:, :-1].columns)
# Create a LIME explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    data_dashboard.iloc[:103, :-1].values, 
    mode="regression",
    training_labels=data_dashboard_y.iloc[:103],
    feature_names=feature_names
)





#LIME *****************************************************************
question = "Choisissez un pays !"  # Définir votre question ici

# Options de la colonne spécifique dans le DataFrame
options = data_dashboard["Country"].unique()  # Supposons que votre colonne s'appelle "Options"

# Liste déroulante pour la sélection de l'option
choix = st.selectbox(question, options,index=list(options).index('Belgium'))

st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
indice = data_dashboard[data_dashboard["Country"] == choix].index[0]

# Select a sample from test data to explain
sample_idx = indice
sample = data_dashboard.iloc[sample_idx,:-1]

#show them as metrics 

value1 = np.round(float(predict_wrapper(sample)[0]),2)
value2 = np.round(data_dashboard_y.iloc[sample_idx].values[0],2)

# Centrer les valeurs dans la colonne
col1, col2 = st.columns(2)
with col1:
    st.metric("QI prédit", value1,delta=f"{np.round(value2-value1,2)}",help="Difference entre le QI prédit le QI réel")

with col2:
    st.metric("QI réel",value2)

st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)
    
# Generate an explanation for the selected sample
exp = explainer.explain_instance(sample, predict_wrapper, num_features=len(feature_names))
custom_css = """
<style>
svg text {
    fill: black !important;
}
svg rect {
    stroke: black !important;
}
svg line {
    stroke: black !important;
}
</style>
"""

# Combine the HTML and custom CSS
html_exp_with_custom_css = custom_css + exp.as_html()
components.html(html_exp_with_custom_css, height=1000)
