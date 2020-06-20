#!/usr/bin/env python
# coding: utf-8

# In[41]:


#Bibliotecas necessárias para projeto
import numpy as np
import pandas as pd
from pandas import DataFrame
import pymysql
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import dask.dataframe as dd
from sqlalchemy import create_engine
import pymysql as pymysql
import sklearn
import pickle
import tkinter as tk
from tkinter import ttk
from random import randint
pio.renderers.default = 'browser'


# In[42]:


#Abre conexao com banco de dados remoto
db_connection_str = 'mysql+pymysql://marianag_dev:senhasecreta@host8.hospedameusite.com.br/marianag_Acidentes_Pos'
db_connection = create_engine(db_connection_str)


# In[43]:


window = tk.Tk()
window.minsize(400, 120)
window.title("Analise de severidade de acidente US")
 
def chosingNumbers():
   window.destroy()
 
label = ttk.Label(window, text = "Escolha um estado")
label.grid(column = 0, row = 0)

amostras = ttk.Label(window, text = "Qtd amostras desejadas")
amostras.grid(column = 0, row = 1)

namostras = tk.StringVar()
cmbobox = ttk.Combobox(window, width = 15, textvariable = namostras)
cmbobox['values'] = ("20","40","60","80","100","120")
cmbobox.grid(column = 1, row = 1)

mynumber = tk.StringVar()
combobox = ttk.Combobox(window, width = 15 , textvariable = mynumber)
combobox['values'] = ("NY","FL","GA","MD","MN","CA")
combobox.grid(column = 1, row = 0)


button = ttk.Button(window, text = "Send", command = chosingNumbers)
button.grid(column = 1, row = 3)
 
 
 
window.mainloop()


# In[44]:


#Carrega modelo
state = mynumber.get()
pkl_filename = '.\\Models\\' + state + ".pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)


# ## Simulação dos inputs

# In[45]:


#Seleciona numero de registros pra simulação
teste_range = int(namostras.get())


# In[46]:


#Busca base dos dados no bando de dados MySql 
query = "SELECT * FROM us_accident_TableTest_"+state+ " ORDER BY RAND() LIMIT  " + str(teste_range)
acc = pd.read_sql(query, con=db_connection)
acidentes_state = acc


# In[47]:


#Prepara dataset de teste e resultados
x_force_teste = acidentes_state.iloc[teste_range-1:teste_range]
simulate = pd.DataFrame(np.repeat(x_force_teste.values,teste_range,axis=0))
simulate.columns = x_force_teste.columns
simulate['Start_Lng'] = acidentes_state['Start_Lng']
simulate['Start_Lat'] = acidentes_state['Start_Lat']
x_force_teste = simulate.copy()
simulate['Prev_Result'] = 0


# In[48]:


#Simulador de valores randomicos
x_force_teste['Duration_Group'] = np.random.randint(1, 11,size = teste_range)
x_force_teste['TemperatureC_Group']  = np.random.randint(1, 7, size = teste_range)
x_force_teste['Distance_Group'] = np.random.randint(1, 10, size = teste_range)


# ## Previsão da severidade

# In[49]:


#Faz as previsoes
y_pred = model.predict(x_force_teste)
#Verifica resultados
y_pred


# In[50]:


x_force_teste['Severity'] = y_pred


# In[51]:


x_result = x_force_teste[['Start_Lat','Start_Lng','Severity']]
x_result.sample(5)


# In[52]:


fig = px.scatter_mapbox(x_result,
                     lat = 'Start_Lat', lon = 'Start_Lng',
                     color = 'Severity', size = 'Severity',
                    color_continuous_scale=px.colors.sequential.Viridis
                    )
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:




