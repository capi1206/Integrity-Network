import re
import networkx as nx
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Group
from dash.dash_table import FormatTemplate
import json

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Integrity Platform'
server = app.server

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

#-------------------------------------------------------------------------------------------------------------------------
#Seccion Transaccional
#extraccion de datos de base smart y creacion del grafo
transaccional = pd.read_excel("C:/Users/difes/BASE OPERACIONES 30-11-2022.xlsx",sheet_name=1)
df_rating=df=pd.read_csv('assets/modeloRat.csv')
df_ci=pd.read_excel('EstructuraDetalladaCIIU_4AC.xls', skiprows=2)
transaccional['producto'] = transaccional['MONTO_APLICACION']*transaccional['PORC_DESC_OP']
df_transaccional = transaccional.groupby(['NOMBRE_EMISOR_OP','NOMBRE_PAGADOR_OP'], as_index=False)['VALOR_NOM_FACT_OP'].sum()
print(df_rating.columns)
#df_red=df_rating[['NIT', 'score', 'Resultado Score']]
result = pd.merge(transaccional, df_rating, left_on='ID_EMISOR_OP', right_on='NIT', how='left')
result2 = pd.merge(result, df_rating, left_on='ID_PAGADOR_OP', right_on='NIT', how='left', suffixes=('', '_pag'))
#formatea los valores inexistentes
result2['score'] = result2['score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'en estudio')
result2['score_pag'] = result2['score_pag'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'en estudio')
result2['Resultado Score'] = result2['Resultado Score'].fillna('')
result2['Resultado Score_pag'] = result2['Resultado Score_pag'].fillna('')
result2['Industria'] = result2['Industria'].fillna('')
result2['Industria_pag'] = result2['Industria_pag'].fillna('')

edges = df_transaccional[['NOMBRE_EMISOR_OP','NOMBRE_PAGADOR_OP']]
#Cuales la manera mas eficiente de desempacar ambas columnas 
#edges se toma de base como pares unicos de emisor-pagador
#veo una manera que es O(n)+O(n)+ hacer una columna con la suma de los strings 
edges = edges.groupby(['NOMBRE_EMISOR_OP','NOMBRE_PAGADOR_OP']).size().reset_index()

edges = edges[['NOMBRE_EMISOR_OP','NOMBRE_PAGADOR_OP']]
transaccional_edges = []

##diccionario de industria a colores
dic_ind_col={'Comercio al por menor (incluso el comercio al por menor de combustibles), excepto el de vehículos automotores y motocicletas':3,
       'Otras industrias manufactureras':3,
       'Elaboración de productos alimenticios':1,
       'Fabricación de productos farmacéuticos, sustancias químicas medicinales y productos botánicos de uso farmacéutico':7,
       'Comercio al por mayor y en comisión o por contrata, excepto el comercio de vehículos automotores y motocicletas':2,
       'Fabricación de productos de caucho y de plástico':2,
       'Confección de prendas de vestir':4,
       'Fabricación de otros productos minerales no metálicos':8,
       'Comercio, mantenimiento y reparación de vehículos automotores y motocicletas, sus partes, piezas y accesorios':6,
       'Actividades de impresión y de producción de copias a partir de grabaciones originales ':5,
       'Fabricación de productos textiles':4,
       'Fabricación de sustancias y productos químicos':8,
       'Fabricación de vehículos automotores, remolques y semirremolques':6,
       'Almacenamiento y actividades complementarias al transporte':6,
       'Fabricación de muebles, colchones y somieres':4,
       'Fabricación de maquinaria y equipo n.c.p.':7,
       'Fabricación de productos elaborados de metal, excepto maquinaria y equipo':8,
       'Fabricación de aparatos y equipo eléctrico':8,
       'Transformación de la madera y fabricación de productos de madera y de corcho, excepto muebles; fabricación de artículos de cestería y espartería':8,
       'Fabricación de productos metalúrgicos básicos':7,
       'Fabricación de papel, cartón y productos de papel y cartón':8,
       'Coquización, fabricación de productos de la refinación del petróleo y actividad de mezcla de combustibles ':6,
       'Instalación, mantenimiento y reparación especializado de maquinaria y equipo':5,
       'Elaboración de bebidas':1,
       'Curtido y recurtido de cueros; fabricación de calzado; fabricación de artículos de viaje, maletas, bolsos de mano y artículos similares, y fabricación de artículos de talabartería y guarnicionería; adobo y teñido de pieles':4,
       'Fabricación de otros tipos de equipo de transporte':6,
       'Fabricación de productos informáticos, electrónicos y ópticos':8,
       'Transporte terrestre; transporte por tuberías':6,
       'Suministro de electricidad, gas, vapor y aire acondicionado ':3,
       'Elaboración de productos de tabaco':1,
       'Correo y servicios de mensajería':3, 'Transporte acuático':6,
       'Transporte aéreo':6, '':0}


for i,j in df_transaccional.iterrows(): #edges.iterrows():
    edge=(j['NOMBRE_EMISOR_OP'],j['NOMBRE_PAGADOR_OP'])
    transaccional_edges.append(edge)

G_trans=nx.from_edgelist(transaccional_edges)

#crear visual
spring_3D = nx.spring_layout(G_trans,dim=3, seed=314,scale = 10)

x_nodes = [spring_3D[i][0] for i in G_trans.nodes()]# x-coordinates of nodes
y_nodes = [spring_3D[i][1] for i in G_trans.nodes()]# y-coordinates
z_nodes = [spring_3D[i][2] for i in G_trans.nodes()]# z-coordinates
edge_list = G_trans.edges()
x_edges=[]
y_edges=[]
z_edges=[]


for edge in edge_list:
    x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
    x_edges += x_coords

    y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
    y_edges += y_coords

    z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
    z_edges += z_coords

hover_text = []
color_nds=[]
for node in G_trans.nodes():
    # Replace 'attribute_name' with the name of the attribute you want to display as the legend
    att_val = result2.loc[result2['NOMBRE_EMISOR_OP'] == node, 'score'].iloc[0] if node in result2['NOMBRE_EMISOR_OP'].values else result2.loc[result2['NOMBRE_PAGADOR_OP'] == node, 'score_pag'].iloc[0]
    result_att =result2.loc[result2['NOMBRE_EMISOR_OP'] == node, 'Resultado Score'].iloc[0] if node in result2['NOMBRE_EMISOR_OP'].values else result2.loc[result2['NOMBRE_PAGADOR_OP'] == node, 'Resultado Score_pag'].iloc[0]
    color_nds.append(result2.loc[result2['NOMBRE_EMISOR_OP'] == node, 'Industria'].iloc[0] if node in result2['NOMBRE_EMISOR_OP'].values else result2.loc[result2['NOMBRE_PAGADOR_OP'] == node, 'Industria_pag'].iloc[0])
    hover_text.append(f'Empresa :{node}<br>Rating: {att_val}<br>   {result_att}')

trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='white', width=1),
                        hoverinfo='none',
                        )
trace_nodes = go.Scatter3d(
    x=x_nodes,
    y=y_nodes,
    z=z_nodes,
    mode='markers',
    marker=dict(
        symbol='circle-open',
        size=10,
        color=[dic_ind_col[ind] for ind in color_nds],      #[random.randint(1,8) for i in G_trans.nodes()],
        colorscale='Viridis',
        opacity=0.8,
        line=dict(color='white', width=1.5)
    ),
    hoverinfo='text',
    text=hover_text,
    textfont=dict(size=14)  # Adjust the font size as needed
)
axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
layout = go.Layout(
                width=1000,
                height=500,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(l=5,r=5,b=5,t=5),
                hovermode='closest',
                )
data = [trace_edges, trace_nodes]

fig_g_transaccional = go.Figure(data=data, layout=layout)
camera = dict(
    up=dict(x=0, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.8, y=0.8, z=0.8)
)
fig_g_transaccional.update_layout(
    scene_camera=camera,
    scene = dict(
                    xaxis = dict(
                        backgroundcolor="rgb(0, 0, 0)",
                        gridcolor="white",
                        showbackground=False,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white")),
                    plot_bgcolor='rgb(0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                  )

#--------------------------------------------------------------------------------------------------------------------------------
#Dashboard derecha de Grafo fig_g_relacional

#--------------------------------------------------------------------------------------------------------------------------

#Valor cartera 

def valor_cartera_unificado(fecha,company_name = None):
    if not company_name:
        df_mask1 = (transaccional['FECHA_RAD_OP'] <= fecha )
        df_mask2 = (transaccional['TIPO_OP'] == 'COMPRA TITULO')
        cuanto_deben = transaccional[df_mask1 & df_mask2]['VALOR_NOM_FACT_OP'].sum()
        #la difrencia entre cuanto me deben y cuanto me han pagado es el valor cartera
        #sumar monto de aplicacion tq fecha de aplicacion sea <= fecha corte y sea una compra titulo
        df_mask3 = (transaccional['FECHA_APLICACION'] <= fecha )
        cuanto_me_han_pagado =  transaccional[df_mask2 & df_mask3]['producto'].sum()
        respuesta =  cuanto_deben-cuanto_me_han_pagado
        return respuesta
    else:
        df_mask1 = (transaccional['FECHA_RAD_OP'] <= fecha )
        df_mask2 = (transaccional['TIPO_OP'] == 'COMPRA TITULO')
        df_mask_pagador = (transaccional['NOMBRE_PAGADOR_OP'] == company_name)
        df_mask_emisor = (transaccional['NOMBRE_EMISOR_OP'] == company_name)
        cuanto_deben = transaccional[df_mask1 & df_mask2 & (df_mask_pagador | df_mask_emisor)]['VALOR_NOM_FACT_OP'].sum()
        #la difrencia entre cuanto me deben y cuanto me han pagado es el valor cartera
        #sumar monto de aplicacion tq fecha de aplicacion sea <= fecha corte y sea una compra titulo
        df_mask3 = (transaccional['FECHA_APLICACION'] <= fecha )
        cuanto_me_han_pagado =  transaccional[df_mask2 & df_mask3 &  (df_mask_pagador | df_mask_emisor)]['producto'].sum()
        respuesta =  cuanto_deben-cuanto_me_han_pagado
        return respuesta
        
fechas = pd.date_range(start='2020-10-31', end='2022-9-30')
data_clean = {'fechas':fechas,
        'valor_cartera':[valor_cartera_unificado(i) for i in fechas]}

data_limpio = pd.DataFrame(data_clean)

fig1 = px.line(data_clean, x="fechas", y='valor_cartera',
    width=500,
    height=300,
    hover_data={"fechas": "|%B %d, %Y"},
    title='Valor cartera')

fig1.update_layout(
    xaxis_title="Fecha", yaxis_title="Valor cartera",
    margin=dict(l=0,r=0,b=0,t=0),
)

cols = [{"name": i, "id": i} 
                for i in df_transaccional.columns]
cols[0]['name'] = 'Razon social Emisor'
cols[1]['name'] = 'Razon social Pagador'
cols[2]['name'] = 'Valor transado'
cols[2]['type'] = 'numeric'
cols[2]['format'] = Format().group(True)

fig2  = dash_table.DataTable(
    id='table222',
    columns=cols,
    data=df_transaccional.to_dict('records'),
    #style_cell=dict(textAlign='left'),
    style_header=dict(backgroundColor="paleturquoise"),
    style_data=dict(backgroundColor="lavender"),
    style_cell={
    # all three widths are needed
    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
    'overflow': 'hidden',
    'textOverflow': 'ellipsis'},
    page_size=5
    )
#------------------------------------------------------------------------------------------------------------------------------
# Grafo relacional

#limpieza de la base 

import json
with open("assets/dic_ent_act.json", "r") as json_file:
    dic_act = json.load(json_file)

bd=pd.read_csv('assets/edges_rel.csv')

df_act=pd.read_excel('assets/entidades_act.xlsx')
df_act.dropna(subset=['ciiu'], inplace=True)

relaciones=[]
for _,j in bd.iterrows():
    relaciones.append((j['n1'], j['n2']))
entidades=list(set([i[0] for i in relaciones]+[i[1] for i in relaciones]))   

#definicion de variables a usar en la generacxion de nuevos enlaces
#lal=[i for i in dic_act]
#ent_f=[e for e in entidades if e in lal]
enlaces_f=[e for e in relaciones if e[0] in dic_act and e[1] in dic_act]
df_rating['ciiu']=df_rating['Actividad '].str[1:].astype(int)

#diccionarios para extraer acividad economica
ent_act_dic=dict(zip(df_act['entidad'],df_act['ciiu']))
dic2=dict(zip(df_rating['nombre'],df_rating['ciiu']))
ent_n=[e for e in dic2]

n_dic={}
for i, j in df_ci.iterrows():
    if isinstance(j['Clase'], float) and not pd.isna(j['Clase']):
        n_dic[int(j['Clase'])]=j['Descripción']
        
    elif isinstance(j['Grupo'], float) and not pd.isna(j['Grupo']):
        n_dic[int(j['Grupo'])]=j['Descripción']

G = nx.Graph()
G.add_edges_from(relaciones)

spring_3D = nx.spring_layout(G, dim=3)

x_nodes,y_nodes,z_nodes = [spring_3D[i][0] for i in G.nodes()],[spring_3D[i][1] for i in G.nodes()],[spring_3D[i][2] for i in G.nodes()]

edge_list = G.edges()
x_edges=[]
y_edges=[]
z_edges=[]


for edge in edge_list:
    x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
    x_edges += x_coords

    y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
    y_edges += y_coords

    z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
    z_edges += z_coords

color = [random.randint(1,8) for i in list(G.nodes())]
trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='white', width=1),
                        hoverinfo='none',
                        )
trace_nodes = go.Scatter3d(x=x_nodes,
                         y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=[random.randint(5,15) for i in list(G.nodes())],
                                    color=color,
                                    # color=1, #color the nodes according to their community
                                    colorscale='Viridis', #either green or mageneta
                                     line=dict(color='black', width=0.5)
                                    ),
                        hoverinfo='text', text = list(G.nodes()))
axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
layout = go.Layout(
                width=1300,
                height=550,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(l=5, r=5, t=5, b=5),
                hovermode='closest',
                )

data = [trace_edges, trace_nodes]
fig_g_relacional = go.Figure(data=data, layout=layout)

camera = dict(
    up=dict(x=0, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.8, y=0.8, z=0.8)
)
fig_g_relacional.update_layout(
                    scene_camera=camera,
                    scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white")),
                    plot_bgcolor='rgb(0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                  )
#-------------------------------------------------------------------------------------------------------------------------------------
# Financial Information

df_financial_information = pd.read_excel("C:/Users/difes/Downloads/ActualizacionInformacion2021 (1).xlsx")
cols_fin = [{"name": i, "id": i} 
                for i in df_financial_information.columns]
cols_fin[1]['name'] = 'Razón social'
cols_fin[2]['name'] = 'Total ingreso operativo (millones)'
cols_fin[2]['type'] = 'numeric'
cols_fin[2]['format'] = Format().group(True)
cols_fin[3]['name'] = 'Ingresos netos por ventas (m)'
cols_fin[3]['type'] = 'numeric'
cols_fin[3]['format'] = Format().group(True)
cols_fin[4]['name'] = 'Utilidad bruta (m)'
cols_fin[4]['type'] = 'numeric'
cols_fin[4]['format'] = Format().group(True)
cols_fin[5]['name'] = 'Margen de ganancia bruta'
cols_fin[5]['type'] = 'numeric'
cols_fin[5]['format'] = Format().group(True)
cols_fin[6]['name'] = 'Ganancia operativa EBIT'
cols_fin[6]['type'] = 'numeric'
cols_fin[6]['format'] = Format().group(True)
cols_fin[7]['name'] = 'Margen operacional'
cols_fin[7]['type'] = 'numeric'
cols_fin[7]['format'] = Format().group(True)
cols_fin[8]['name'] = 'Margen operacional'
cols_fin[8]['type'] = 'numeric'
cols_fin[8]['format'] = Format().group(True)
fig_financial  = dash_table.DataTable(
    id='table',
    columns=cols_fin,
    data=df_financial_information.to_dict('records'),
    #style_cell=dict(textAlign='left'),
    style_header=dict(backgroundColor="paleturquoise"),
    style_data=dict(backgroundColor="lavender"),
    style_cell_conditional=[
        {
            'if': {'column_id': 'razon_Social'},
            'textAlign': 'left'
        }
    ],
    # style_cell={
    # # all three widths are needed
    # 'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
    # 'overflow': 'hidden',
    # 'textOverflow': 'ellipsis'},
    filter_action='native',
    page_size=20
    )
#--------------------------------------------------------------------------------------------------------------------------------
#Ranking Information
#git test
df_ranking = pd.read_excel("C:/Users/difes/Downloads/IntegrityPlatform-master (1)/IntegrityPlatform-master/data/Integrity Ranking.xlsx")
percentage = FormatTemplate.percentage(2)
cols_ranking = [{"name": i, "id": i} 
                for i in df_ranking.columns]
cols_ranking[3]['type'] = 'numeric'
cols_ranking[3]['format'] =percentage
cols_ranking[4]['type'] = 'numeric'
cols_ranking[4]['format'] =percentage
cols_ranking[5]['type'] = 'numeric'
cols_ranking[5]['format'] = Format().group(True)
fig_ranking  = dash_table.DataTable(
    id='table_rank',
    columns=cols_ranking,
    data=df_ranking.to_dict('records'),
    #style_cell=dict(textAlign='left'),
    style_header=dict(backgroundColor="paleturquoise"),
    style_data=dict(backgroundColor="lavender"),
    #Compañia
    style_cell_conditional=[
        {
            'if': {'column_id': 'Compañia'},
            'textAlign': 'left'
        }
    ],
    filter_action='native',
    page_size=20
    )


#functions
#------------------------------------------------------------------------------------------------------------------------------------
def in_out_l(nm, l):
    return 1 if nm in l else 0

#función que predice si hay relación nodo1->nodo2  
def enlace(ent1,ent2):
    k=2
    mx=100
    min_d=mx
    min_d2=mx
    act1=dic_act[ent1] if ent1 in dic_act else dic2[ent1]
    act2=dic_act[ent2] if ent2 in dic_act else dic2[ent2]
    if ent1 not in dic_act:
        min_ind=0
        for i in dic_act:
            acti=dic_act[i]
            d_aux=abs(acti -act1)
            if d_aux <3 and d_aux < min_d:
                min_ind=i
                min_d=d_aux
        ent1=min_ind
        if min_ind==0:
            return 0
    if ent2 not in dic_act:
        min_ind=0
        for i in dic_act:
            acti=dic_act[i]
            d_aux=abs(acti -act2)
            if d_aux <3 and d_aux < min_d:
                min_ind=i
                min_d2=d_aux
        ent2=min_ind
        if min_ind==0:
            return 0
    act1=dic_act[ent1]
    act2=dic_act[ent2]    
    if min_d <= min_d2:
        out_list=[k[1] for k in enlaces_f if k[0]==ent1]
        dic_dist_ent={}
        for j in dic_act:
            if j!=ent2:
                dic_dist_ent[j]=abs(act2-dic_act[j])
        sorted_distances = sorted(dic_dist_ent.items(), key=lambda x: x[1])[:k] 
        sorted_distances.append(ent2)
        return sum([in_out_l(n, out_list) for n in sorted_distances])/(k+1)
    else:
        in_list=[k[0] for k in enlaces_f if k[1]==ent2]
        dic_dist_ent={}
        for j in dic_act:
            if j!=ent1:
                dic_dist_ent[j]=abs(act1-dic_act[j])
        sorted_distances = sorted(dic_dist_ent.items(), key=lambda x: x[1])[:k] 
        sorted_distances.append(ent1)
        return sum([in_out_l(n, in_list) for n in sorted_distances])/(k+1)




def main_financials_modal(company_clicked):
    result = df_financial_information.loc[df_financial_information['razon_Social'] == company_clicked]

    if len(result) == 0:
      return ['Not found']*12
    else:
      response = []
      listof = list(df_financial_information.columns)[2:]
      listof.remove('activos_totales')
      listof.remove('activos_corrientes')
      listof.remove('pasivos_totales')
      listof.remove('dueda_neta')
      listof.remove('flujo_neto_de_efectivo_activo')
      for i in listof:
        response.append(result.iloc[0][i])
      return response



def describe_insights(company_clicked):
    result = df_ranking.loc[df_ranking['Compañia'] == company_clicked]
    return [result.iloc[0]['Proveedores Potenciales'], result.iloc[0]['Clientes Potenciales'],"{:,}".format(int(result.iloc[0]['Integrity Ranking 1%'])) ,int(1000*result.iloc[0]['Eigen'])]
    
def zoomed_graph(node_clicked,degree_neighbors = 2):
    # selected_nodes = list(G[node_clicked])+[node_clicked]
    N = nx.ego_graph(G=G,n=node_clicked,radius=degree_neighbors)
    spring_3D = nx.spring_layout(N,dim=3, seed=2719,scale = 50)

    x_nodes = [spring_3D[i][0] for i in N.nodes()]# x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in N.nodes()]# y-coordinates
    z_nodes = [spring_3D[i][2] for i in N.nodes()]# z-coordinates
    edge_list = N.edges()
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords


    size=[random.randint(25,40) for i in list(N.nodes())]
    color = [random.randint(0, 8) for i in list(N.nodes())]
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            line=dict(color='white', width=3),
                            hoverinfo='none')
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers+text',
                            textfont=dict(color='#E6F4F1',size=15),
                            marker=dict(
                                symbol='circle',
                                size=size,
                                color= color,
                                colorscale='Viridis'),
                                 #color the nodes according to their community
                                # colorscale=['lightgreen','magenta'], #either green or mageneta
                                #line=dict(color='black', width=0.5)),
                            hoverinfo='text', 
                            text=list(N.nodes()) 
                                ) 
                                
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    layout = go.Layout(
                    width=1300,
                    height=550,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),  
                    margin=dict(l=5, r=5, t=5, b=5),
                    hovermode='closest')

    data = [trace_edges, trace_nodes]
    zoom_figure = go.Figure(data=data, layout=layout)

    zoom_figure.update_layout( hoverlabel_font_color='rgb(255,255,255)',
                    scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(0, 0, 0)",
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(0, 0,0)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white")),
                    plot_bgcolor='rgb(0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                  )
    return zoom_figure



def zoom_graph_trans(node_clicked,degree_neighbors = 2):
    N = nx.ego_graph(G=G_trans,n=node_clicked,radius=degree_neighbors)
    spring_3D = nx.spring_layout(N,dim=3, seed=314,scale = 20)

    x_nodes = [spring_3D[i][0] for i in N.nodes()]# x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in N.nodes()]# y-coordinates
    z_nodes = [spring_3D[i][2] for i in N.nodes()]# z-coordinates
    edge_list = N.edges()
    x_edges=[]
    y_edges=[]
    z_edges=[]

    for edge in edge_list:
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords

        



    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            line=dict(color='white', width=3),
                            hoverinfo='none',
                            )
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers+text',
                            textfont=dict(color='#E6F4F1',size=15),
                            marker=dict(symbol='circle-open',
                                        size=15,
                                        color=[random.randint(0, 5) for i in N.nodes()],
                                        colorscale='Viridis', #either green or mageneta
                                        # line=dict(color='black', width=0.5)
                                        ),
                            hoverinfo='text', text = list(N.nodes()))
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    layout = go.Layout(
                    width=1000,
                    height=500,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                    margin=dict(l=5,r=5,b=5,t=5),
                    hovermode='closest',
                    )

    data = [trace_edges, trace_nodes]
    fig_g_transaccional = go.Figure(data=data, layout=layout)

    fig_g_transaccional.update_layout(scene = dict(
                        xaxis = dict(
                            backgroundcolor="rgb(0, 0, 0)",
                            gridcolor="white",
                            showbackground=False,
                            zerolinecolor="white",),
                        yaxis = dict(
                            backgroundcolor="rgb(0, 0,0)",
                            gridcolor="white",
                            showbackground=False,
                            zerolinecolor="white"),
                        zaxis = dict(
                            backgroundcolor="rgb(0, 0,0)",
                            gridcolor="white",
                            showbackground=False,
                            zerolinecolor="white")),
                        plot_bgcolor='rgb(0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
    return fig_g_transaccional



#working capital, la cuenta integrity, credito corporativo, recaudos y pagos, commodo wallet 
#  Callbacks
#---------------------------------------------------------------------------------------------------------------------------------
@app.callback(
    [Output('relational_graph','figure'),
    Output('search_bar','value'),
    Output('accordion-contents-financial', "children"),
    Output('accordion-contents-product-recomendation', "children"),
    Output('visible_financials','style'),
    Output('visible_recommendation','style'),
    Output('progress-working-capital','value'),
    Output("progress-working-capital", "label"),
    Output('progress-collections','value'),
    Output("progress-collections", "label"),
    Output('progress-integrity_account','value'),
    Output("progress-integrity_account", "label"),
    Output('progress-corp_credit','value'),
    Output("progress-corp_credit", "label"),
    Output('progress-commodo','value'),
    Output("progress-commodo", "label"),
    Output("potential_suppliers", "children"),
    Output("potential_clients", "children"),
    Output("integrity_ranking", "children"),
    Output("Eigen_rank", "children"),
    Output("total_ingreso_operativo", "children"),
    Output("ingresos_netos_por_ventas", "children"),
    Output("utilidad_bruta", "children"),
    Output("margen_ganancia_bruta", "children"),
    Output("ganancia_operativa_EBIT", "children"),
    Output("margen_operacional", "children"),
    Output("EBIDTA", "children"),
    Output("margen_ebidta", "children"),
    Output("ganancia_perdida_neta", "children"),
    Output("margen_neto", "children"),
    Output("total_empleados", "children"),
    Output("CIUU", "children")],
    [Input('search_bar','value'),
    Input('relational_graph', 'clickData')],
     prevent_initial_call=True
)
def barra_actualiza_estado(search,clicks):
    callback_id = ctx.triggered_id
    if callback_id =='relational_graph':
        click_dict = clicks['points'][0]
        if 'text' in click_dict.keys():
            zoom_figure = zoomed_graph(click_dict['text'])
            title_despliege= 2*[f"Company selected: {click_dict['text']}"]
            visibles = 2*[{'display': 'block'}]
            random_list = random.sample(range(10, 100), 5)
            unpack_progress = [x for i in random_list for x in (i,f"{i} %")]
            unpack_insights = describe_insights(click_dict['text'])
            unpack_main_financials = main_financials_modal(click_dict['text'])
            result = [zoom_figure, click_dict['text']]+title_despliege+visibles+unpack_progress+unpack_insights+unpack_main_financials
            return  result

        else:
            raise PreventUpdate
    if callback_id =='search_bar':
        if search == None:
            return [fig_g_relacional, dash.no_update, "Select a company...", "Select a company...", {'display': 'none'}, {'display': 'none'}]+14*[dash.no_update]+12*[dash.no_update]
        else:
            zoom_figure = zoomed_graph(search)
            random_list = random.sample(range(10, 100), 5)
            unpack_progress = [x for i in random_list for x in (i,f"{i} %")]
            unpack_insights = describe_insights(search)
            unpack_main_financials = main_financials_modal(search)
            return [zoom_figure, dash.no_update, f"Company selected: {search}", f"Company selected: {search}", {'display': 'block'}, {'display': 'block'}]+unpack_progress+unpack_insights+unpack_main_financials
    return 32*[dash.no_update]



def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


app.callback(
    Output("modal-financials", "is_open"),
    Input("financials-modal-button", "n_clicks"),
    State("modal-financials", "is_open"),
    prevent_initial_call=True
)(toggle_modal)


@app.callback(
    Output("n-rel-table", "style"),  # Add this output for hiding/showing the table
    Output("n-rel-data-table", "data"),  # Add this output for updating the table data
    [Input("gen-rel-button", "n_clicks")],
)
def gen_n_relaciones(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        print("Encontrando nuevos enlaces...")

        l_nodo,l_actividad, l_nodo2,l_actividad2=[],[],[],[]
        ent_f=[f for f in dic_act]
        lee=len(ent_f)
        l2=len(ent_n)
        for _ in range(100000):
    
            if random.random() >0.4999:
                e1=ent_f[int(random.random()*lee)]
            else:
                e1=ent_n[int(random.random()*l2)]
            if random.random() >0.4999:
                e2=ent_f[int(random.random()*lee)]
            else:
                e2=ent_n[int(random.random()*l2)]    
    
            if e1!=e2 and (e1, e2) not in enlaces_f:
                sc=enlace(e1,e2)
                if sc >0:
                    act1=dic_act[e1] if e1 in ent_act_dic else dic2[e1]
                    act2=dic_act[e2] if e2 in ent_act_dic else dic2[e2]
                    l_nodo.append(e1)
                    l_nodo2.append(e2)
                    l_actividad.append(n_dic[act1])
                    l_actividad2.append(n_dic[act2])
        print('finalizó')
        d_aux=pd.DataFrame(  {'nodo1':l_nodo,
                   'actividad1':l_actividad,
                   'nodo2':l_nodo2,
                   'actividad2':l_actividad2,})
        #d_aux.to_excel('nuevas relaciones.xlsx', index=False)
        table_style={  "overflow": "auto", "max-width": "100%"}
        
        # Update the data of the table with d_aux
        table_data = d_aux.to_dict("records")
    else:
        table_style = {"display": "none"}
        table_data = []
    return table_style, table_data

@app.callback(
    Output('valor_cartera_pie_chart','figure'),
    Output('transactional_graph','figure'),
    Input('search_bar_trans', 'value'),
    prevent_initial_call = True
)
def trans_search_updates_cartera(company_name):
    if company_name == None:
        return fig1,fig_g_transaccional
    else:
        data_clean = {'fechas':fechas,
            'valor_cartera':[valor_cartera_unificado(i,company_name) for i in fechas]}
        data_limpio = pd.DataFrame(data_clean)
        fig1_especifica = px.line(data_clean, x="fechas", y='valor_cartera',
            width=500,
            height=300,
            hover_data={"fechas": "|%B %d, %Y"},
        title='Valor cartera - '+company_name)
        fig1_especifica.update_traces(line_color='#ff0000', line_width=5)

        return fig1_especifica,zoom_graph_trans(company_name)

@app.callback(
    [Output('test_api_balance','children')],
    [Input('api_status_button','n_clicks',)],
    [State('search_bar','value')],
    prevent_initial_call = True
)
def api_balance_test(clicks,search):
    #return api data, search inside api most similar
    return ['Current search "{}" and number of clicks "{}"'.format(
        search,
        clicks
    )]

card_1 = dbc.Card(
    [
        dbc.CardImg(src="/assets/financials-1.jpg", top=True),
        dbc.CardBody([
            html.P("Potential suppliers:", className="card-text"),
            html.P("Potential 2:",id='potential_suppliers', className="card-text"),
            html.Div([
            dbc.Button("Additional info", color="info")],
            className="d-grid gap-2")]
        ),
    ],
    style={"width": "15rem"},
    color="primary", 
    outline=True
)
card_2 = dbc.Card(
    [
        dbc.CardImg(src="/assets/financials-2.jpg", top=True),
        dbc.CardBody([
            html.P("Potential clients:", className="card-text"),
            html.P("Potential 2:",id='potential_clients', className="card-text"),
            html.Div([
            dbc.Button("Additional info", color="info")],
            className="d-grid gap-2"),
        ]
        ),
    ],
    style={"width": "15rem"},
    color="primary", 
    outline=True
)

card_3 = dbc.Card(
    [
        dbc.CardImg(src="/assets/financials-3.jpg", top=True),
        dbc.CardBody([
            html.P("Integrity Ranking:", className="card-text"),
            html.P("Potential 2:",id='integrity_ranking', className="card-text"),
            html.Div([
            dbc.Button("Additional info", color="info")],
            className="d-grid gap-2")
        ]
        ),
    ],
    style={"width": "15rem"},
    color="primary", 
    outline=True
)

card_4 = dbc.Card(
    [
        dbc.CardImg(src="/assets/financials-4.jpg", top=True),
        dbc.CardBody(
            [
            html.P("Eigen Rank:", className="card-text"),
            html.P("Potential 2:",id='Eigen_rank', className="card-text"),
            html.Div([
            dbc.Button("Additional info", color="info")],
            className="d-grid gap-2")
            ]   
        ),
    ],
    style={"width": "15rem"},
    color="primary", 
    outline=True
)


cards = dbc.Row(
    [
        dbc.Col(card_1,width="auto"),
        dbc.Col(card_2,width="auto"),
        dbc.Col(card_3,width="auto"),
        dbc.Col(card_4,width="auto")
    ],
    justify="center"
)

#------------------------------------------------------------------------------------------------------------------------------
#Recomendation 
compatibility =  html.Div(
    [
        html.Th("Working capital",id = 'working-capital',style={"textDecoration": "underline", "cursor": "pointer"}),
        dbc.Tooltip('''El capital de trabajo neto le dice cuánto dinero tiene disponible para cubrir los gastos actuales.
        El resultado puede ser una cifra positiva o negativa y se utiliza para determinar el perfil de riesgo financiero de una empresa.
        Un capital de trabajo positivo indica que una empresa puede hacer frente a sus deudas y está en condiciones de facilitar el crecimiento,
        mientras que un capital de trabajo negativo indica que la empresa puede tener problemas.''',
            target='working-capital',
        ),
        dbc.Progress(id = 'progress-working-capital',value=25, color="success", className="mb-3",striped=True, animated =True, label = "25%"),
        html.Th("Collections and payments"),
        dbc.Progress(id = 'progress-collections', value=50, color="warning", className="mb-3",striped=True, animated =True,label = "50%"),
        html.Th("Integrity account"),
        dbc.Progress(id = 'progress-integrity_account',value=75, color="danger", className="mb-3",striped=True, animated =True,label = "75%"),
        html.Th("Corporate credit"),
        dbc.Progress(id = 'progress-corp_credit',value=95, color="secondary",className="mb-3", striped=True, animated =True,label = "95%"),
        html.Th("Commodo wallet"),
        dbc.Progress(id ='progress-commodo', value=67, color="info",className="mb-3", striped=True, animated =True, label = "67%"),
    ]
)

#------------------------------------------------------------------------------------------------------------------------------------------
# Tables for specific financials

table_header_principal = [
    html.Thead(html.Tr([html.Th("Main financial ratios"), html.Th("2021"),html.Th("2020")]))
]




row1_principal = html.Tr([html.Td("Total ingreso operativo"), html.Td("Na",id='total_ingreso_operativo'),html.Td("Na",id='total_ingreso_operativo_2020')])
row2_principal = html.Tr([html.Td("Ingresos netos por ventas"), html.Td("Na",id='ingresos_netos_por_ventas'),html.Td("Na",id='ingresos_netos_por_ventas_2020')])
row3_principal = html.Tr([html.Td("Utilidad bruta"), html.Td("Na",id = 'utilidad_bruta'),html.Td("Na",id = 'utilidad_bruta_2020')])
row4_principal = html.Tr([html.Td("Margen Ganancia bruta"), html.Td("Na",id='margen_ganancia_bruta'),html.Td("Na",id='margen_ganancia_bruta_2020')])
row5_principal = html.Tr([html.Td("Ganancia operativa EBIT"), html.Td("Na",id='ganancia_operativa_EBIT'),html.Td("Na",id='ganancia_operativa_EBIT_2020')])
row6_principal = html.Tr([html.Td("Margen operacional"), html.Td("Na",id = 'margen_operacional'),html.Td("Na",id = 'margen_operacional_2020')])
row7_principal = html.Tr([html.Td("EBITDA"), html.Td("Na",id = 'EBIDTA'),html.Td("Na",id = 'EBIDTA_2020')])
row8_principal = html.Tr([html.Td("Margen EBITDA"), html.Td("Na",id='margen_ebidta'), html.Td("Na",id='margen_ebidta_2020')])
row9_principal = html.Tr([html.Td("Ganancia (perdida) neta"), html.Td("Na",id = 'ganancia_perdida_neta'), html.Td("Na",id = 'ganancia_perdida_neta_2020')])
row10_principal = html.Tr([html.Td("Margen neto"), html.Td("Na",id = 'margen_neto'), html.Td("Na",id = 'margen_neto_2020')])
row11_principal = html.Tr([html.Td("Total_empleados"), html.Td("Na",id = 'total_empleados'), html.Td("Na",id = 'total_empleados_2020')])
row12_principal = html.Tr([html.Td("CIUU"), html.Td("Na",id = 'CIUU'),html.Td("Na",id = 'CIUU_2020')])

table_body_principal = [html.Tbody([row1_principal, row2_principal, row3_principal, row4_principal,
                                row5_principal,row6_principal,row7_principal,row8_principal,row9_principal,row10_principal,row11_principal,
                                row12_principal])]

table_modal_principal = dbc.Table(table_header_principal+table_body_principal , bordered=True)




table_header_balance = [
    html.Thead(html.Tr([html.Th("Partidas del balance (COP)"), html.Th("Valor")]))
]

row1 = html.Tr([html.Td("Efectivo y equivalentes al efectivo"), html.Td("Na")])
row2 = html.Tr([html.Td("Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes: "), html.Td("Na")])
row3 = html.Tr([html.Td("Inventarios corrientes"), html.Td("Na")])
row4 = html.Tr([html.Td("Activos corrientes"), html.Td("Na")])
row5 = html.Tr([html.Td("Activos no corrientes"), html.Td("Na")])
row6 = html.Tr([html.Td("Activos totales"), html.Td("Na")])
row7 = html.Tr([html.Td("Cuentas por pagar"), html.Td("Na")])
row8 = html.Tr([html.Td("Dueda financiera"), html.Td("Na")])
row9 = html.Tr([html.Td("Pasivos corrientes"), html.Td("Na")])
row10 = html.Tr([html.Td("Pasivos No corrientes"), html.Td("Na")])
row11 = html.Tr([html.Td("Pasivos totales"), html.Td("Na")])
row12 = html.Tr([html.Td("Total patrimonio"), html.Td("Na")])

table_body_balance = [html.Tbody([row1, row2, row3, row4,
                                row5,row6,row7,row8,row9,row10,row11,
                                row12])]

table_modal_balance = dbc.Table(table_header_balance + table_body_balance, bordered=True)

table_header_resultados = [
    html.Thead(html.Tr([html.Th("Partidas del Estado de resultados (COP)"), html.Th("Valor")]))
]

row1_resultados = html.Tr([html.Td("Ingresos brutos"), html.Td("Na")])
row2_resultados = html.Tr([html.Td("Costos"), html.Td("Na")])
row3_resultados = html.Tr([html.Td("Utilidad bruta"), html.Td("Na")])
row4_resultados = html.Tr([html.Td("Gastos administrativos"), html.Td("Na")])
row5_resultados = html.Tr([html.Td("Gastos ventas y distribucion"), html.Td("Na")])
row6_resultados = html.Tr([html.Td("Otros gastos"), html.Td("Na")])
row7_resultados = html.Tr([html.Td("Otros ingresos"), html.Td("Na")])
row8_resultados = html.Tr([html.Td("Utilidad operativa (EBIT)"), html.Td("Na")])
row9_resultados = html.Tr([html.Td("EBIDTA"), html.Td("Na")])
row10_resultados = html.Tr([html.Td("Ingresos financieros "), html.Td("Na")])
row11_resultados = html.Tr([html.Td("Gastos financieros"), html.Td("Na")])
row12_resultados = html.Tr([html.Td("Ingresos no operativos"), html.Td("Na")])
row13_resultados = html.Tr([html.Td("Egresos no operativos"), html.Td("Na")])
row14_resultados = html.Tr([html.Td("Utilidad antes de impuestos"), html.Td("Na")])
row15_resultados = html.Tr([html.Td("Impuestos"), html.Td("Na")])
row16_resultados = html.Tr([html.Td("Utilidad neta"), html.Td("Na")])





table_body_resultados = [html.Tbody([row1_resultados, row2_resultados, row3_resultados, row4_resultados,
                                row5_resultados,row6_resultados,row7_resultados,row8_resultados,row9_resultados,
                                row10_resultados,
                                row11_resultados,
                                row12_resultados,
                                row13_resultados,
                                row14_resultados,
                                row15_resultados,
                                row16_resultados])]

table_modal_resultados = dbc.Table(table_header_resultados + table_body_resultados, bordered=True)


modal = html.Div(
    [
    dbc.Button('Financials',id= 'financials-modal-button',color= 'info'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle('Financial Information')),
            table_modal_principal,
            html.Br(),
            dbc.Button("Load finacials", color="secondary",
                     className="me-1",id ='api_status_button',
                     size ='sm' ),
            html.P(id='test_api_balance'),
            table_modal_balance,
            html.Br(),
            table_modal_resultados
        ],
        id = 'modal-financials',
        size = 'lg',
        is_open = False
    )
    ],
    className="d-grid gap-2"
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Alert("Select a company...",
                    color="info",
                    id ='accordion-contents-product-recomendation'),
                    html.Div(
                        id='visible_recommendation',
                        children = [
                        compatibility
                        ],
                        style= {'display':'none'}
                    )
                ],
                title="Product recommendation",
            ),
            dbc.AccordionItem(
                [
                    dbc.Alert("Select a company...",
                    color="info",
                    id ='accordion-contents-financial'),
                    html.Div(
                        id ='visible_financials',
                        children=[
                            modal,
                            html.Br(),
                            cards],
                        style= {'display': 'none'})
                ],
                title="Business insights",
            )
        ],
        start_collapsed=True
    )
)

row_smart =dbc.Row(
    [
        dbc.Col(                    dcc.Graph(id= 'valor_cartera_pie_chart',
                        figure=fig1),width="auto"),
        dbc.Col(fig2,width="auto"),
        # card_1,
        # card_2,
        # card_3,
        # card_4
    ],
    justify="center"
)


accordion_smart = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Alert("Select a company...",
                    color="info",
                    id ='alert_smart'),
                    row_smart

                ],
                title="Analytics",
            ),
        ],
        start_collapsed=True
    )
)
butt_style={"background-color": "#007BFF",  # Background color
        "color": "white",  # Text color
        "border": "none",  # Remove button border
        "border-radius": "5px",  # Add rounded corners
        "padding": "10px 20px",  # Adjust padding
        "font-size": "16px",  # Adjust font size
        "cursor": "pointer",  
}
app_style={
        "background-color": "rgb(14,19,31)",  # Set the background color to black
        "height": "100vh",  # Set the height to cover the entire viewport
    }

#-------------------------------------------------------------------------------------------------------------------------------------------
#app layout
app.layout = html.Div(style=app_style,
    children=[
    html.Img(src=app.get_asset_url('integrity.png'),style={'height':'10%', 'width':'25%'}),
    dcc.Tabs([
        #Tab #1 Relational Graph
        dcc.Tab(label='Relational Graph', children=[
            dcc.Dropdown(options = list(G.nodes()),
                placeholder="Select a company",
                multi=False,
                id='search_bar'),
            html.Button("Generar nuevas relaciones", id="gen-rel-button", style=butt_style),    
            html.Div([
                html.Center([
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children= dcc.Graph(
                            id= 'relational_graph',
                            figure=fig_g_relacional,
                            config= {'displaylogo': False},
                            style={"border":"2px black solid"}),
                            fullscreen=False)]),
            accordion
             ]),
             html.Div(
    id="n-rel-table",  # This is the ID to reference in the callback
    children=[
        dash_table.DataTable(
            id="n-rel-data-table",
            columns=[
                {"name": "Nodo 1", "id": "nodo1"},
                {"name": "Actividad 1", "id": "actividad1"},
                {"name": "Nodo 2", "id": "nodo2"},
                {"name": "Actividad 2", "id": "actividad2"},
            ],
            style_table={'overflowY': 'scroll', 'maxHeight': '80vh'}, 
        )
    ],
    style={"display": "none", "overflowX": "auto", "max-width": "100%", "max-height": "80vh"}
             )
    ]),
        #Transactional Graph
        dcc.Tab(label='Transactional Graph', children=[
            dcc.Dropdown(options = list(G_trans.nodes()),
                placeholder="Select a company...",
                multi=False,
                id='search_bar_trans'),
            html.Div([
                html.Center(
                    dcc.Graph(id= 'transactional_graph',
                        figure=fig_g_transaccional,
                        config= {'displaylogo': False}, 
                        style={'display': 'inline-block'})
                ),
                        ],
                    ),
                    accordion_smart]),
        dcc.Tab(label='Financial Information', children=[
            fig_financial]),
        dcc.Tab(label='Integrity Ranking', children=[
            fig_ranking
        ])
    ],)
])

if __name__ == '__main__':
    app.run_server(debug=True)
    