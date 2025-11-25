# Efeito do Microclima e do Manejo na Taxa de Sobrevivência de Mudas 

<p align="center">
  <a href="https://github.com/Edu-png">
    <img src="https://img.shields.io/badge/Autor-Eduardo%20Coqueiro-purple?style=flat&logo=github" alt="Autor">
  </a>
  <a href="mailto:eduardocoqueiro@gmail.com">
    <img src="https://img.shields.io/badge/Email-eduardocoqueiro%40gmail.com-purple?style=flat&logo=gmail" alt="Email">
  </a>
  <a href="https://linkedin.com/in/eduardocoqueiro/">
    <img src="https://img.shields.io/badge/LinkedIn-Eduardo%20Coqueiro-purple?style=flat&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="https://kaggle.com/EduardoCoqueiro">
    <img src="https://img.shields.io/badge/Kaggle-Eduardo%20Coqueiro-blue?style=flat&logo=kaggle" alt="Kaggle">
  </a>
</p>
<img width="1920" height="1080" alt="CAPAS - PROJETOS (1)" src="https://github.com/user-attachments/assets/187fd527-2970-4c6e-ad64-04930f726057" />

## 📚 Sumário

1. [☀️ Introdução](#️-introdução)  
   - [🎯 Objetivo](#-objetivo)
2. [🔄 Pipeline do Projeto](#-pipeline-do-projeto)  
3. [🧪 Metodologia](#-metodologia)
   - [1. Coleta e Tratamento de Dados](#1-coleta-e-tratamento-de-dados)
   - [2. Análise Exploratória de Dados (EDA)](#2-análise-exploratória-de-dados-eda)
   - [3. Engenharia de Atributos](#3-engenharia-de-atributos)
   - [4. Balanceamento com SMOTE](#4-balanceamento-com-smote)
   - [5. Modelagem Supervisionada](#5-modelagem-supervisionada)
   - [6. Interpretabilidade com Explainable-AI](#6-interpretabilidade-com-explainable-ai)
   - [7. Comparação e Conclusão](#7-comparação-e-conclusão)
4. [📊 Resultados e Conclusões](#-resultados-e-conclusões)
   - [🎯 Distribuição da Variável-Alvo](#-distribuição-da-variável-alvo)
   - [🌿 Distribuição das Variáveis Numéricas](#-distribuição-das-variáveis-numéricas)
   - [🔍 Boxplots Gerais das Variáveis](#-boxplots-gerais-das-variáveis)
   - [🔗 Matriz de Correlação](#-matriz-de-correlação)
   - [🌡️ Efeito das Variáveis de Microclima](#-efeito-das-variáveis-de-microclima)
   - [🌳 Taxa de Sobrevivência por Gênero](#-taxa-de-sobrevivência-por-gênero)
   - [🧭 Relação entre Distância e Sobrevivência](#-relação-entre-distância-e-sobrevivência)
   - [🤖 Comparação de Modelos Preditivos](#-comparação-de-modelos-preditivos)
   - [🧩 Conclusões Gerais](#-conclusões-gerais)
5. [🧩 Considerações Finais](#-considerações-finais)
   - [Pontos de Destaque](#pontos-de-destaque)
   - [Limitações e Melhorias](#limitações-e-melhorias)
   - [Próximos Passos](#próximos-passos)
6. [📞 Contato](#-contato)


## ☀️ Introdução

A sobrevivência de mudas florestais é influenciada por uma série de fatores microclimáticos e práticas de manejo realizadas durante o plantio. Entender como variáveis como sombreamento, espécies companheiras, cobertura do solo, atraso no plantio e distância do centro de plantio afetam a taxa de sobrevivência pode gerar insights valiosos para aumentar a eficiência operacional e reduzir custos com replantio.

Este projeto aplica uma abordagem analítica e preditiva para identificar padrões nos dados, quantificar o impacto de variáveis-chave e construir modelos de machine learning interpretáveis, promovendo uma tomada de decisão mais informada por parte de pesquisadores e gestores florestais.

### 🎯 Objetivo

O objetivo principal é desenvolver modelos preditivos que expliquem e antecipem a taxa de sobrevivência de mudas com base em fatores ambientais e de manejo, com foco em interpretabilidade. Além disso, pretende-se:

- Compreender os principais fatores associados à mortalidade de mudas;
- Explorar o comportamento das variáveis numéricas e categóricas envolvidas;
- Avaliar o desempenho de modelos como Random Forest, XGBoost e Regressão Logística;
- Utilizar técnicas de Explainable AI para comunicar os achados de forma clara e visual.

## 🔄 Pipeline do Projeto

A seguir está o fluxo geral seguido no projeto para análise da taxa de sobrevivência de mudas:

1. **Importação e Tratamento dos Dados**
   - Leitura da base original de plantio.
   - Seleção de variáveis de interesse.
   - Filtragem de registros incompletos ou inconsistentes.

2. **Análise Exploratória (EDA)**
   - Análise da distribuição da variável-alvo (sobrevivência).
   - Exploração visual das variáveis numéricas e categóricas.
   - Correlações e padrões iniciais.

3. **Balanceamento da Base**
   - Aplicação do algoritmo SMOTE para lidar com desbalanceamento entre classes (mortalidade vs. sobrevivência).

4. **Pré-processamento e Feature Engineering**
   - Normalização de variáveis contínuas.
   - Codificação de variáveis categóricas.
   - Criação de novas variáveis como *distância do centro*, indicadores de sombra e presença de espécies companheiras.

5. **Modelagem**
   - Treinamento de diferentes algoritmos (Logistic Regression, Random Forest, XGBoost).
   - Avaliação por meio de métricas como acurácia, recall e F1-score.
   - Validação cruzada e análise comparativa de desempenho.

6. **Interpretabilidade (XAI)**
   - Aplicação de ferramentas como SHAP para explicar o impacto das variáveis.
   - Geração de gráficos de importância global e local das features.

7. **Conclusões e Recomendações**
   - Identificação dos principais fatores ligados à mortalidade.
   - Sugestões para práticas de manejo com base nas variáveis mais relevantes.

## 🧪 Metodologia

A metodologia aplicada neste projeto foi cuidadosamente desenhada para garantir a robustez estatística, a interpretabilidade dos modelos e a relevância prática dos insights gerados. As etapas a seguir descrevem o processo adotado do início ao fim da análise.

---

### 1. Coleta e Tratamento de Dados  
- Importação de uma base contendo informações sobre condições ambientais, geográficas e operacionais de plantios florestais.  
- Remoção de colunas irrelevantes e filtragem de dados com base na completude das informações.  
- Padronização e renomeação de variáveis para facilitar a análise e garantir consistência.

---

### 2. Análise Exploratória de Dados (EDA)  
- Inspeção da distribuição da variável-alvo `survived`, revelando desbalanceamento significativo.  
- Análise da distribuição das variáveis numéricas: `banana`, `panicum`, `cedrela`, `cover`, `delay`, `dist_center`.  
- Análise da influência do gênero das mudas (`genus`) na sobrevivência.  
- Avaliação da correlação entre variáveis contínuas para evitar multicolinearidade.

---

### 3. Engenharia de Atributos  
- Criação da variável `dist_center` com base em coordenadas para medir distância do centro do plantio.  
- Conversão da variável-alvo para o tipo categórico binário (`0 = morreu`, `1 = sobreviveu`).  
- Aplicação de normalização nas variáveis numéricas.  
- Codificação das variáveis categóricas via one-hot encoding.

---

### 4. Balanceamento com SMOTE  
- Aplicação da técnica **SMOTE (Synthetic Minority Oversampling Technique)** para corrigir o desbalanceamento da variável `survived`, garantindo que os modelos não sejam enviesados para a classe majoritária.  
- Comparação do desempenho dos modelos com e sem a aplicação do SMOTE.

---

### 5. Modelagem Supervisionada  
Três algoritmos foram utilizados e comparados:

- **Regressão Logística**: como baseline interpretável.  
- **Random Forest**: modelo de ensemble baseado em árvores de decisão, robusto para dados mistos.  
- **XGBoost**: modelo gradualmente otimizado com excelente desempenho em tarefas tabulares.

Cada modelo foi avaliado por:

- Acurácia  
- Precisão  
- Recall  
- F1-Score  
- Curva ROC (área sob a curva)

---

### 6. Interpretabilidade com Explainable AI (XAI)  
- Utilização da biblioteca `SHAP` para explicar a influência de cada variável nas predições do modelo.  
- Geração de gráficos de importância global (summary plots) e explicações locais (force plots) para casos específicos.  
- Visualização da variação do impacto de cada variável sobre a previsão da sobrevivência de uma muda.

---

### 7. Comparação e Conclusão  
- Foram comparados os desempenhos dos modelos antes e depois do balanceamento com SMOTE.  
- As variáveis mais importantes foram identificadas com base nos modelos e nas explicações de SHAP, fornecendo diretrizes para práticas de manejo mais eficazes.


## 📊 Resultados e Conclusões

Nesta seção, são apresentados os principais resultados obtidos a partir da análise exploratória dos dados e do treinamento dos modelos preditivos, com ênfase nos fatores de microclima e manejo que influenciam a **taxa de sobrevivência de mudas**.

---

### 🎯 Distribuição da variável-alvo

<img width="647" height="404" alt="1" src="https://github.com/user-attachments/assets/a8803710-7739-4689-a711-424e49f6901a" />

A base apresenta **desequilíbrio significativo entre classes**, com **88,3% das mudas mortas (0)** e apenas **11,7% sobreviventes (1)**.  
Esse desbalanceamento impôs desafios para a modelagem, exigindo o uso de técnicas de **reclassificação e balanceamento (SMOTE)** para evitar vieses durante o treinamento.

---

### 🌿 Distribuição das variáveis numéricas

<img width="1384" height="815" alt="2" src="https://github.com/user-attachments/assets/d4a01883-a35d-4efc-b0e9-d099638d1435" />

As variáveis relacionadas ao **microclima local** apresentaram padrões distintos:
- **`cover` (cobertura vegetal)** concentra valores altos, indicando áreas com maior sombreamento;
- **`cedrela`** e **`panicum`**, associadas à vegetação concorrente, mostraram distribuições assimétricas, com predominância de valores baixos;
- **`dist_center`** (distância ao centro da parcela) segue distribuição quase normal, com leve concentração entre 50 e 70 metros;
- **`delay`** (tempo até a coleta) mostrou maior variação, podendo refletir diferenças no manejo entre locais de plantio.

---

### 🔍 Boxplots gerais das variáveis

<img width="1384" height="584" alt="3" src="https://github.com/user-attachments/assets/099422e7-ec86-4e53-88da-8c4b6b779951" />

Os boxplots indicam **alta dispersão** em variáveis contínuas, com presença de *outliers* em `cedrela` e `cover`.  
Destaca-se que as variáveis de cobertura e competição vegetal apresentam **comportamentos opostos**, sugerindo que maior sombreamento tende a proteger as mudas, enquanto vegetação competidora influencia negativamente na sobrevivência.

---

### 🔗 Matriz de correlação

<img width="668" height="537" alt="6" src="https://github.com/user-attachments/assets/e7333034-692b-4aaa-869b-6c83d6b5645d" />

As correlações entre variáveis numéricas foram em geral **baixas a moderadas**.  
A única relação negativa expressiva foi entre **`cedrela` e `cover` (-0,37)**, reforçando a hipótese de competição entre vegetação nativa e cobertura protetora.  
Esse resultado indica **baixa multicolinearidade**, favorecendo a utilização conjunta dessas variáveis em modelos supervisionados.

---

### 🌡️ Efeito das variáveis de microclima na sobrevivência

<img width="1784" height="982" alt="5" src="https://github.com/user-attachments/assets/f17381ce-c49a-4b02-966a-52a4c95c038e" />

Os boxplots comparando o status das mudas (vivas e mortas) mostraram:
- **Maior sombreamento (`cover`)** está associado a **maior taxa de sobrevivência**;  
- **Altos valores de `panicum` e `cedrela`** (espécies competidoras) correlacionam-se com **maior mortalidade**;  
- **`delay` e `dist_center`** tiveram influência mais discreta, sugerindo efeito indireto no desempenho.

Essas relações indicam que o **microclima e o manejo da vegetação acompanhante** são determinantes na fase inicial de estabelecimento.

---

### 🌳 Taxa de sobrevivência por gênero

<img width="1011" height="653" alt="4" src="https://github.com/user-attachments/assets/d47f62a1-7503-4693-946c-a91012187344" />

A sobrevivência varia amplamente entre gêneros botânicos.  
Os maiores valores foram observados para **Celtis (44,4%)**, seguido de **Nesogordonia (25,7%)** e **Mansonia (25,3%)**.  
Espécies como **Gambeya**, **Pycnanthus** e **Piptadeniastrum** apresentaram taxas inferiores a 1%.  
Isso evidencia **forte influência genética**, apontando a necessidade de manejo diferenciado e seleção de genótipos adaptados ao ambiente local.

---

### 🧭 Relação entre distância e sobrevivência

<img width="705" height="479" alt="7" src="https://github.com/user-attachments/assets/6dafbc81-9e5d-4d93-9055-b2ebc08b50d5" />

Mudas mais próximas do **centro da parcela** tendem a apresentar **maior sobrevivência**, possivelmente devido à **melhor uniformidade microclimática** e **redução de efeitos de borda**.  
Esse resultado reforça a importância do **planejamento espacial no plantio** e do controle de variações ambientais dentro das áreas experimentais.

---

### 🤖 Comparação de modelos preditivos

<img width="984" height="584" alt="8" src="https://github.com/user-attachments/assets/641ae42d-288d-41ed-9a6d-ab64980e04f6" />

Foram testados diferentes algoritmos supervisionados: **XGBoost**, **Random Forest**, **Regressão Logística**, e um **ensemble com balanceamento SMOTE**.  
Os resultados mostraram:

| Modelo | Acurácia | Recall | F1-score | Observações |
|:-------|:---------:|:-------:|:---------:|:------------|
| **XGBoost** | **0.89** | 0.78 | 0.62 | Melhor desempenho geral, bom equilíbrio entre precisão e recall. |
| **Random Forest** | 0.88 | **0.80** | 0.61 | Melhor recall, mas menor precisão. |
| **Logistic Regression** | 0.87 | 0.62 | 0.50 | Desempenho inferior, porém interpretável. |
| **Ensemble SMOTE + Threshold** | 0.89 | 0.77 | 0.61 | Solução mais balanceada para o desbalanceamento de classes. |

O **XGBoost** apresentou o melhor desempenho geral, destacando-se pela **robustez e capacidade de generalização**, mesmo em dados desbalanceados.  
A combinação com técnicas de oversampling (SMOTE) aprimorou a **identificação de sobreviventes**, sem perda relevante de acurácia.

---

### 🧩 Conclusões gerais

Os resultados evidenciam que:

- **Fatores de microclima e manejo**, especialmente **cobertura vegetal e competição por luz**, são **determinantes para o sucesso do plantio**;  
- O **sombramento adequado** atua como **fator protetor**, reduzindo a mortalidade inicial;  
- Há **forte variação entre gêneros**, indicando a necessidade de **seleção genética adaptada** e estratégias diferenciadas de implantação;  
- Modelos baseados em **árvores de decisão e boosting** oferecem **excelente poder preditivo**, permitindo a **identificação precoce de áreas de risco**.

Em síntese, o estudo demonstra o potencial da **inteligência artificial aplicada ao manejo florestal**, fornecendo **insights explicáveis e acionáveis** para otimizar a **taxa de sobrevivência de mudas** e aprimorar práticas de restauração ecológica.

---

## 🧩 Considerações Finais

O estudo apresentou uma abordagem integrada entre **análise de dados ambientais**, **modelagem preditiva** e **interpretação de resultados**, evidenciando o impacto do microclima e do manejo na sobrevivência de mudas.  
Os resultados obtidos confirmam a relevância de variáveis como **sombreamento (cover)** e **competição vegetal (panicum e cedrela)** como fatores determinantes para o sucesso inicial das mudas em campo.

A análise revelou que:
- **O microclima exerce influência direta** sobre a taxa de sobrevivência, especialmente em condições de maior cobertura vegetal;
- **A diversidade genética entre gêneros** implica respostas distintas à competição e às condições ambientais;
- O uso de **modelos interpretáveis de Machine Learning**, como **XGBoost** e **Random Forest**, mostrou-se eficiente para capturar essas relações complexas;
- Técnicas de **balanceamento de classes (SMOTE)** foram fundamentais para garantir previsões mais justas e reduzir o viés de mortalidade predominante.

A partir dos modelos desenvolvidos, é possível prever áreas e condições com **maior risco de mortalidade**, permitindo ações de manejo mais precisas e direcionadas, como **ajustes no sombreamento**, **controle de espécies competidoras** e **seleção de genótipos mais adaptados**.

Em síntese, o trabalho demonstra que a **integração entre ciência de dados e ecologia florestal** é uma ferramenta poderosa para **otimizar processos de restauração e produção de mudas**, contribuindo para práticas de manejo mais **eficientes, sustentáveis e baseadas em evidências**.

---

📘 **Próximos passos:**
1. Implementar explicabilidade via **SHAP values** para identificar a contribuição individual de cada variável nas previsões.  
2. Expandir o modelo com **dados climáticos sazonais** e **parâmetros de solo** para refinar as análises.  
3. Validar o modelo em novos experimentos de campo e diferentes regiões de plantio, ampliando sua **capacidade preditiva e generalização**.

---

🧠 *Este projeto reforça o papel da ciência de dados como aliada da pesquisa florestal, fornecendo subsídios quantitativos para tomadas de decisão mais assertivas e sustentáveis no manejo de ecossistemas e viveiros.*

<div align="center">
  <img src="https://github.com/user-attachments/assets/54afb33c-97be-40b6-8c96-0f12852e946f" alt="thank-you" width="500">
</div>

## 📞 Contato
- **LinkedIn:** [Eduardo Coqueiro](https://www.linkedin.com/in/eduardocoqueiro/)
- **Site:** [Eduardo Coqueiro](https://dataguy.my.canva.site/eduardo-coqueiro)
- **Kaggle:** [Eduardo Coqueiro](https://www.kaggle.com/eduardocoqueiro)
