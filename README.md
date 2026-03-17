# Datathon Fase 5

## 📌 Objetivo
O objetivo deste Datathon é desenvolver uma proposta preditiva a partir da análise dos dados 
disponibilizados pela ONG Passos Mágicos, assumindo o papel de cientista de dados.

## 📊 Dados
A base de dados contém informações educacionais e socioeconômicas dos estudantes da Passos 
Mágicos no período de 2022, 2023 e 2024. Foram disponibilizadas duas bases de dados com as 
características de desenvolvimento educacional e questões socioeconômicas dos estudantes e 
um dicionário de dados com o mapeamento de todas as variáveis. Além da base de dados, alguns 
relatórios de pesquisa realizada pela Passos Mágicos também foram disponibilizados para 
auxiliar no conhecimento do negócio. 

## 🛠 Tecnologias
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Jupyter Notebook

## 📁 Estrutura do Projeto
- data/: dados brutos e tratados
- notebooks/: análises exploratórias
- src/: códigos organizados
- model/: modelos treinados

## 🚀 Resultados
A análise dos dados educacionais da Associação Passos Mágicos entre os anos de 2022 e 2024 permitiu identificar padrões relevantes relacionados ao desempenho e à evolução dos estudantes ao longo do programa. Observou-se um crescimento no número de alunos atendidos e uma redução progressiva nos níveis mais severos de defasagem educacional, que passaram de 3,3% em 2022 para 0,3% em 2024. Esses resultados sugerem uma evolução positiva no desempenho educacional dos participantes.

Além disso, as análises indicaram que indicadores como engajamento (IEG) e desempenho acadêmico (IDA) apresentam forte relação com o desenvolvimento educacional dos alunos, contribuindo significativamente para a variação do Índice de Desenvolvimento Educacional (INDE). Esses fatores se destacam como elementos importantes para acompanhar a trajetória dos estudantes e compreender seu progresso ao longo das diferentes fases do programa.

Na etapa preditiva, foi desenvolvido um modelo utilizando o algoritmo XGBoost, com o objetivo de estimar a probabilidade de um aluno entrar em situação de risco de defasagem educacional. O modelo apresentou acurácia de 0,777 e AUC de 0,85, indicando boa capacidade de classificação e de distinção entre alunos com maior ou menor probabilidade de risco.

Para facilitar a aplicação prática da solução, foi desenvolvida uma aplicação web utilizando Streamlit, permitindo inserir informações dos estudantes e obter estimativas de risco em tempo real. Dessa forma, a análise de dados e o modelo preditivo podem apoiar a ONG na identificação precoce de alunos que necessitam de acompanhamento, contribuindo para a tomada de decisões e para o fortalecimento das estratégias educacionais da organização.
