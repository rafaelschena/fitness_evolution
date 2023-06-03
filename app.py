import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import datetime

# Define a function to load the dataset from CSV
@st.cache_data
def load_dataset():
    df = pd.read_csv('dados_nutri.csv')
    df['Data'] = df['Data'].astype('datetime64')
    return df

# Define a function to update the dataset by adding a row
def add_row(dataset, new_row):
    dataset = dataset.append(new_row, ignore_index=True)
    return dataset

# Check if the dataset is already loaded in the session state
if 'df' not in st.session_state:
    # Load the dataset from CSV and store it in the session state
    st.session_state.df = load_dataset()


st.title('Avaliações físicas - Evolução')

tab1, tab2, tab3, tab4 = st.tabs(['🔢 Dados', '📉 Peso', '💪 Musculo', '🫃 Gordura'])

# global df
# df = df_orig

with tab1:
    st.title('Dados das informações nutricionais')
    #global df

    data = st.date_input("Data da avaliação:", datetime.date.today())
    peso = st.number_input("Digite o peso:")
    musc_perc = st.number_input("Digite o percentual de músculo:")
    musc_kg = peso * musc_perc * 0.01
    fat_perc = st.number_input("Digite o percentual de gordura:")
    fat_kg = peso * fat_perc * 0.01

    if st.button('Inserir dados'):
        #st.write(f'Tipo da data: {type(data)}; Tipo do peso: {type(peso)}, colunas: {df.columns}')
        # 'Data', 'Peso_kg', 'Massa_magra_percent', 'Massa_magra_kg', 'Gordura_percent', 'Gordura_kg', 'Musculo_percent', 'Musculo_kg', 'Gord_visceral_ind', 'Circ_abdom_cm', 'Braco_esq_cm', 'Braço_dir_cm', 'Perna_esq_cm', 'Perna_dir_cm', 'Pant_esq_cm', 'Pant_dir_cm', 'Torax_cm', 'Cintura_cm', 'Abdomen_cm', 'Quadril_cm'
        st.session_state.df = st.session_state.df.append({'Data': data, 'Peso_kg': peso, 'Massa_magra_percent': np.nan, 'Massa_magra_kg': np.nan,
                        'Gordura_percent': fat_perc, 'Gordura_kg': fat_kg, 'Musculo_percent': musc_perc, 'Musculo_kg': musc_kg,
                        'Gord_visceral_ind': np.nan, 'Circ_abdom_cm': np.nan, 'Braco_esq_cm': np.nan, 'Braço_dir_cm': np.nan,
                        'Perna_esq_cm': np.nan, 'Perna_dir_cm': np.nan, 'Pant_esq_cm': np.nan, 'Pant_dir_cm': np.nan,
                        'Torax_cm': np.nan, 'Cintura_cm': np.nan, 'Abdomen_cm': np.nan, 'Quadril_cm': np.nan}, ignore_index=True)
        st.session_state.df['Data'] = st.session_state.df['Data'].astype('datetime64')

        st.subheader(f'Dados inseridos em {data}:')
        st.text(f'Peso (kg): {peso}')
        st.text(f'Percentual de músculo: {musc_perc}')
        st.text(f'Massa muscular (kg): {musc_kg}')
        st.text(f'Percentual de gorgura: {fat_perc}')
        st.text(f'Massa de gordura (kg): {fat_kg}')


        

with tab2:
    col = 'Peso_kg'
    df_passado = st.session_state.df[['Data', col]]
    df_passado['Data_base'] = df_passado['Data'].iloc[0]
    df_passado['dias_corridos'] = df_passado['Data'] - df_passado['Data_base']
    df_passado['dias_corridos'] = df_passado['dias_corridos'].dt.days

    X_train = df_passado['dias_corridos'].values.reshape(-1, 1)
    y_train = df_passado[col].values.reshape(-1, 1)

    LinReg = LinearRegression().fit(X_train, y_train)

    X_future = np.arange(X_train.take(-1), 365, 30).reshape(-1, 1)
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

        X_future = np.arange(X_train.take(-1), 365, 30).reshape(-1, 1)
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

        X_future = np.arange(X_train.take(-1), 365, 30).reshape(-1, 1)
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

        X_future = np.arange(X_train.take(-1), 365, 30).reshape(-1, 1)
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

        X_future = np.arange(X_train.take(-1), 365, 30).reshape(-1, 1)
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