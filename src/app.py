import streamlit as st
import pandas as pd
import joblib
import os


# Caminho do diretório onde app.py está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho do modelo
caminho_modelo = os.path.join(BASE_DIR, "../model/modelo_risco.pkl")

st.title("Modelo XGBoost - Métricas")

# Valores manuais
accuracy = 0.7773109243697479
roc_auc = 0.85

st.write(f"**Acurácia:** {accuracy:.4f}")
st.write(f"**Curva ROC (AUC):** {roc_auc:.2f}")

# Carrega o modelo
if os.path.exists(caminho_modelo):
    model = joblib.load(caminho_modelo)

else:
    model = None


st.title("📊 Predição de Risco de Defasagem Escolar")

st.write("Insira os indicadores do aluno para prever o risco de defasagem.")

# Inputs do usuário
IAA = st.number_input("IAA - Indicador de AutoAvaliação", 0.0, 10.0)
IEG = st.number_input("IEG - Indicador de Engajamento", 0.0, 10.0)
IPS = st.number_input("IPS - Indicador Psicossocial", 0.0, 10.0)
IDA = st.number_input("IDA - Indicador de Aprendizagem", 0.0, 10.0)
IPV = st.number_input("IPV - Indicador de Ponto de Virada", 0.0, 10.0)
IPP = st.number_input("IPP - Indicador Psicopedagogico", 0.0, 10.0)
Idade = st.number_input("Idade", 0, 35)
#Instituicao = st.selectbox("Instituição de Ensino", [0,1,2])
Ano_ingresso = st.number_input("Ano de ingresso", 2018, 2025)
#Genero = st.selectbox("Gênero", [0,1])

#Gênero Formatado para o usuário
genero_opcoes = {"Feminino": 0, "Masculino": 1}
Genero_nome = st.selectbox("Gênero", list(genero_opcoes.keys()))
Genero = genero_opcoes[Genero_nome]

#Instituição Formatado para o usuário
instituicao_opcoes = {
    "Pública": 0,
    "Privada": 1,
    "Sem Classificação": 2
}
Instituicao_nome = st.selectbox("Instituição de Ensino", list(instituicao_opcoes.keys()))
Instituicao = instituicao_opcoes[Instituicao_nome]

if st.button("Prever risco"):

    dados = pd.DataFrame({
        "IAA":[IAA],
        "IEG":[IEG],
        "IPS":[IPS],
        "IDA":[IDA],
        "IPV":[IPV],
        "IPP":[IPP],
        "Idade":[Idade],
        "Instituicao_bin":[Instituicao],
        "Ano ingresso":[Ano_ingresso],
        "Genero_bin":[Genero]
    })

    prob = model.predict_proba(dados)[0][1]

    st.subheader(f"Probabilidade de Defasagem: {prob:.2f}")

    if prob < 0.3:
        st.success("Baixo risco")
    elif prob < 0.7:
        st.warning("Risco moderado")
    else:
        st.error("Alto risco")