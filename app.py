import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import datetime
import gspread
import os
import json

# Define a function to load the dataset from CSV
@st.cache_data
def load_dataset():
    df = pd.read_csv('dados_nutri.csv')
    df['Data'] = df['Data'].astype('datetime64')
    return df

@st.cache_data
def load_spreadsheet():
    gspread_credentials_str = os.getenv("GSPREAD_CREDENTIALS")
    # gspread_credentials_str = st.secrets("GSPREAD_CREDENTIALS")
    gspread_credentials_dic = json.loads(gspread_credentials_str)
    gc = gspread.service_account_from_dict(gspread_credentials_dic)
    sheet_url = os.getenv("GSPREAD_URL")
    # print(sheet_url)
    pasta = gc.open_by_url(sheet_url)
    planilha = pasta.worksheet('dados_nutri')

    colunas = planilha.get_all_values().pop(0)
    df = pd.DataFrame(data=planilha.get_all_values(), columns=colunas)
    df.drop(index=0, inplace=True)
    df.reset_index(inplace=True)
    df['Data'] = df['Data'].astype('datetime64')

    return df

def add_row_spreadsheet(new_data_list):
    gspread_credentials_str = os.getenv("GSPREAD_CREDENTIALS")
    # gspread_credentials_str = st.secrets("GSPREAD_CREDENTIALS")
    gspread_credentials_dic = json.loads(gspread_credentials_str)
    gc = gspread.service_account_from_dict(gspread_credentials_dic)
    sheet_url = os.getenv("GSPREAD_URL")
    pasta = gc.open_by_url(sheet_url)
    planilha = pasta.worksheet('dados_nutri')
    planilha.append_row(values=new_data_list)

# Define a function to update the dataset by adding a row
def add_row(dataset, new_row):
    dataset = dataset.append(new_row, ignore_index=True)
    return dataset

# Check if the dataset is already loaded in the session state
if 'df' not in st.session_state:
    # Load the dataset from CSV and store it in the session state
    # st.session_state.df = load_dataset()
    st.session_state.df = load_spreadsheet()


st.title('Avaliações físicas - Evolução')

tab1, tab2, tab3, tab4 = st.tabs(['🔢 Dados', '📉 Peso', '💪 Musculo', '🫃 Gordura'])

# global df
# df = df_orig

with tab1:
    st.title('Dados das informações nutricionais')
    #global df

    data = st.date_input("Data da avaliação:", datetime.date.today())
    peso = st.number_input("Digite o peso (kg):")
    massa_magra_perc = st.number_input("Digite o percentual de massa magra:")
    massa_magra_kg = peso * massa_magra_perc * 0.01
    fat_perc = st.number_input("Digite o percentual de gordura:")
    fat_kg = peso * fat_perc * 0.01
    musc_perc = st.number_input("Digite o percentual de músculo:")
    musc_kg = peso * musc_perc * 0.01
    gord_visc_ind = st.number_input("Digite o índice de gordura visceral:")
    
    braco_esq_cm = st.number_input("Digite a circunferência do braço esquerdo (cm):")
    braco_dir_cm = st.number_input("Digite a circunferência do braço direito (cm):")
    perna_esq_cm = st.number_input("Digite a circunferência da perna esquerda (cm):")
    perna_dir_cm = st.number_input("Digite a circunferência da perna direita (cm):")
    pant_esq_cm = st.number_input("Digite a circunferência da panturrilha esquerda (cm):")
    pant_dir_cm = st.number_input("Digite a circunferência da panturrilha direita (cm):")
    torax_cm = st.number_input("Digite a circunferência do tórax (cm):")
    cintura_cm = st.number_input("Digite a circunferência da cintura (cm):")
    circ_abdom_cm = st.number_input("Digite a circunferência do abdômen (cm):")
    quadril_cm = st.number_input("Digite a circunferência do quadril (cm):")


    if st.button('Inserir dados'):
        #st.write(f'Tipo da data: {type(data)}; Tipo do peso: {type(peso)}, colunas: {df.columns}')
        # 'Data', 'Peso_kg', 'Massa_magra_percent', 'Massa_magra_kg', 'Gordura_percent', 'Gordura_kg', 'Musculo_percent', 'Musculo_kg', 'Gord_visceral_ind', 'Circ_abdom_cm', 'Braco_esq_cm', 'Braço_dir_cm', 'Perna_esq_cm', 'Perna_dir_cm', 'Pant_esq_cm', 'Pant_dir_cm', 'Torax_cm', 'Cintura_cm', 'Abdomen_cm', 'Quadril_cm'
        new_data_dic = {'Data': data, 'Peso_kg': peso, 'Massa_magra_percent': massa_magra_perc, 'Massa_magra_kg': massa_magra_kg,
                        'Gordura_percent': fat_perc, 'Gordura_kg': fat_kg, 'Musculo_percent': musc_perc, 'Musculo_kg': musc_kg,
                        'Gord_visceral_ind': gord_visc_ind, 'Circ_abdom_cm': circ_abdom_cm, 'Braco_esq_cm': braco_esq_cm, 'Braço_dir_cm': braco_dir_cm,
                        'Perna_esq_cm': perna_esq_cm, 'Perna_dir_cm': perna_dir_cm, 'Pant_esq_cm': pant_esq_cm, 'Pant_dir_cm': pant_dir_cm,
                        'Torax_cm': torax_cm, 'Cintura_cm': cintura_cm, 'Abdomen_cm': circ_abdom_cm, 'Quadril_cm': quadril_cm}
        new_data_list = list(new_data_dic.values())[1:]
        new_data_list.insert(0, str(data))

        add_row_spreadsheet(new_data_list)        

        st.session_state.df = st.session_state.df.append(new_data_dic, ignore_index=True)
        st.session_state.df['Data'] = st.session_state.df['Data'].astype('datetime64')

        st.subheader(f'Dados inseridos em {data}:')
        st.text(f'Peso (kg): {peso}')
        st.text(f'Percentual de massa magra: {massa_magra_perc}')
        st.text(f'Massa magra (kg): {massa_magra_kg}')
        st.text(f'Percentual de gorgura: {fat_perc}')
        st.text(f'Massa de gordura (kg): {fat_kg}')        
        st.text(f'Percentual de músculo: {musc_perc}')
        st.text(f'Massa muscular (kg): {musc_kg}')
        st.text(f'Índice de gordura visceral: {gord_visc_ind}')
        st.text(f'Braço esquerdo (cm): {braco_esq_cm}')
        st.text(f'Braço direito (cm): {braco_dir_cm}')
        st.text(f'Perna esquerda (cm): {perna_esq_cm}')
        st.text(f'Perna direita (cm): {perna_dir_cm}')
        st.text(f'Panturilha esquerda (cm): {pant_esq_cm}')
        st.text(f'Panturilha direita (cm): {pant_dir_cm}')
        st.text(f'Tórax (cm): {torax_cm}')
        st.text(f'Cintura (cm): {cintura_cm}')
        st.text(f'Abdômen (cm): {circ_abdom_cm}')
        st.text(f'Quadril (cm): {quadril_cm}')
        
        



        

with tab2:
    col = 'Peso_kg'
    df_passado = st.session_state.df[['Data', col]]
    df_passado['Data_base'] = df_passado['Data'].iloc[0]
    df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
    df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

    X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
    y_train = df_passado[col].values.reshape(-1, 1)

    LinReg = LinearRegression().fit(X_train, y_train)

    X_future = np.arange(X_train.take(-1), 365*2, 30).reshape(-1, 1)
    y_future = LinReg.predict(X_future)

    df_future = pd.DataFrame()
    df_future['delta_d'] = X_future.squeeze()
    df_future['delta_d'] = pd.to_timedelta(df_future['delta_d'], unit='D')
    df_future['Data_base'] = df_passado['Data'].loc[0]
    df_future['Data'] =  df_future['Data_base'] + df_future['delta_d']
    df_future['Previsao'] = y_future
    df_future.drop(columns=['delta_d', 'Data_base'], inplace=True)

    df_passado.drop(columns=['dias_corridos', 'Data_base'], inplace=True)

    df_merged = pd.merge(df_passado, df_future, on='Data', how='outer')

    trace1 = go.Scatter(x=df_merged['Data'], y=df_merged[col], name=col,
                        mode='lines', line=dict(color='red', width=2))
    trace2 = go.Scatter(x=df_merged['Data'], y=df_merged['Previsao'], name='Previsão',
                        mode='lines', line=dict(color='blue', width=2, dash='dot'))

    layout = go.Layout(title=f'{col} e previsão futura', xaxis=dict(title='Data'), yaxis=dict(title=col))

    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with tab3:

    col1, col2 = st.columns(2)

    with col1:

        col = 'Musculo_percent'
        df_passado = st.session_state.df[['Data', col]]
        df_passado['Data_base'] = df_passado['Data'].iloc[0]
        df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
        df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

        X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
        y_train = df_passado[col].values.reshape(-1, 1)


        LinReg = LinearRegression().fit(X_train, y_train)

        X_future = np.arange(X_train.take(-1), 365*2, 30).reshape(-1, 1)
        y_future = LinReg.predict(X_future)

        df_future = pd.DataFrame()
        df_future['delta_d'] = X_future.squeeze()
        df_future['delta_d'] = pd.to_timedelta(df_future['delta_d'], unit='D')
        df_future['Data_base'] = df_passado['Data'].loc[0]
        df_future['Data'] =  df_future['Data_base'] + df_future['delta_d']
        df_future['Previsao'] = y_future
        df_future.drop(columns=['delta_d', 'Data_base'], inplace=True)

        df_passado.drop(columns=['dias_corridos', 'Data_base'], inplace=True)

        df_merged = pd.merge(df_passado, df_future, on='Data', how='outer')

        trace1 = go.Scatter(x=df_merged['Data'], y=df_merged[col], name=col,
                            mode='lines', line=dict(color='red', width=2))
        trace2 = go.Scatter(x=df_merged['Data'], y=df_merged['Previsao'], name='Previsão',
                            mode='lines', line=dict(color='blue', width=2, dash='dot'))

        layout = go.Layout(title=f'{col} e previsão futura', xaxis=dict(title='Data'), yaxis=dict(title=col))

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    with col2:
        col = 'Musculo_kg'
        df_passado = st.session_state.df[['Data', col]]
        df_passado['Data_base'] = df_passado['Data'].iloc[0]
        df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
        df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

        X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
        y_train = df_passado[col].values.reshape(-1, 1)

        LinReg = LinearRegression().fit(X_train, y_train)

        X_future = np.arange(X_train.take(-1), 365*2, 30).reshape(-1, 1)
        y_future = LinReg.predict(X_future)

        df_future = pd.DataFrame()
        df_future['delta_d'] = X_future.squeeze()
        df_future['delta_d'] = pd.to_timedelta(df_future['delta_d'], unit='D')
        df_future['Data_base'] = df_passado['Data'].loc[0]
        df_future['Data'] =  df_future['Data_base'] + df_future['delta_d']
        df_future['Previsao'] = y_future
        df_future.drop(columns=['delta_d', 'Data_base'], inplace=True)

        df_passado.drop(columns=['dias_corridos', 'Data_base'], inplace=True)

        df_merged = pd.merge(df_passado, df_future, on='Data', how='outer')

        trace1 = go.Scatter(x=df_merged['Data'], y=df_merged[col], name=col,
                            mode='lines', line=dict(color='red', width=2))
        trace2 = go.Scatter(x=df_merged['Data'], y=df_merged['Previsao'], name='Previsão',
                            mode='lines', line=dict(color='blue', width=2, dash='dot'))

        layout = go.Layout(title=f'{col} e previsão futura', xaxis=dict(title='Data'), yaxis=dict(title=col))

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)

with tab4:
    col3, col4 = st.columns(2)

    with col3:
        col = 'Gordura_percent'
        df_passado = st.session_state.df[['Data', col]]
        df_passado['Data_base'] = df_passado['Data'].iloc[0]
        df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
        df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

        X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
        y_train = df_passado[col].values.reshape(-1, 1)

        LinReg = LinearRegression().fit(X_train, y_train)

        X_future = np.arange(X_train.take(-1), 365*2, 30).reshape(-1, 1)
        y_future = LinReg.predict(X_future)

        df_future = pd.DataFrame()
        df_future['delta_d'] = X_future.squeeze()
        df_future['delta_d'] = pd.to_timedelta(df_future['delta_d'], unit='D')
        df_future['Data_base'] = df_passado['Data'].loc[0]
        df_future['Data'] =  df_future['Data_base'] + df_future['delta_d']
        df_future['Previsao'] = y_future
        df_future.drop(columns=['delta_d', 'Data_base'], inplace=True)

        df_passado.drop(columns=['dias_corridos', 'Data_base'], inplace=True)

        df_merged = pd.merge(df_passado, df_future, on='Data', how='outer')

        trace1 = go.Scatter(x=df_merged['Data'], y=df_merged[col], name=col,
                            mode='lines', line=dict(color='red', width=2))
        trace2 = go.Scatter(x=df_merged['Data'], y=df_merged['Previsao'], name='Previsão',
                            mode='lines', line=dict(color='blue', width=2, dash='dot'))

        layout = go.Layout(title=f'{col} e previsão futura', xaxis=dict(title='Data'), yaxis=dict(title=col))

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    with col4:
        col = 'Gordura_kg'
        df_passado = st.session_state.df[['Data', col]]
        df_passado['Data_base'] = df_passado['Data'].iloc[0]
        df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
        df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

        X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
        y_train = df_passado[col].values.reshape(-1, 1)

        LinReg = LinearRegression().fit(X_train, y_train)

        X_future = np.arange(X_train.take(-1), 365*2, 30).reshape(-1, 1)
        y_future = LinReg.predict(X_future)

        df_future = pd.DataFrame()
        df_future['delta_d'] = X_future.squeeze()
        df_future['delta_d'] = pd.to_timedelta(df_future['delta_d'], unit='D')
        df_future['Data_base'] = df_passado['Data'].loc[0]
        df_future['Data'] =  df_future['Data_base'] + df_future['delta_d']
        df_future['Previsao'] = y_future
        df_future.drop(columns=['delta_d', 'Data_base'], inplace=True)

        df_passado.drop(columns=['dias_corridos', 'Data_base'], inplace=True)

        df_merged = pd.merge(df_passado, df_future, on='Data', how='outer')

        trace1 = go.Scatter(x=df_merged['Data'], y=df_merged[col], name=col,
                            mode='lines', line=dict(color='red', width=2))
        trace2 = go.Scatter(x=df_merged['Data'], y=df_merged['Previsao'], name='Previsão',
                            mode='lines', line=dict(color='blue', width=2, dash='dot'))

        layout = go.Layout(title=f'{col} e previsão futura', xaxis=dict(title='Data'), yaxis=dict(title=col))

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        st.plotly_chart(fig, theme='streamlit', use_container_width=True)