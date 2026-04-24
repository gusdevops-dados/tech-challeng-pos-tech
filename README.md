# 📊 Análise de Reclamações e Impacto no NPS

## 🎯 Objetivo do Projeto

O objetivo deste projeto é analisar o impacto das reclamações de clientes sobre o NPS (Net Promoter Score), investigando se clientes que registram reclamações possuem maior propensão a serem detratores.

Além disso, o projeto busca:

* Identificar variáveis relevantes para previsão de detratores
* Validar estatisticamente a relação entre reclamações e satisfação do cliente
* Construir um modelo preditivo para classificação de clientes
* Gerar insights acionáveis para melhoria da experiência do cliente

---

## 📁 Descrição da Base de Dados

A base de dados utilizada contém informações de clientes e suas interações, incluindo:

* Variáveis relacionadas ao comportamento do cliente
* Indicadores de reclamação
* Métrica de satisfação (NPS)

### Principais variáveis:

* `reclamacao` → indica se o cliente registrou reclamação
* `nps` → score de satisfação do cliente
* Demais variáveis explicativas utilizadas no modelo

A variável alvo do modelo foi construída a partir do NPS, classificando clientes como:

* **Detratores**
* **Neutros**
* **Promotores**

Para fins de modelagem, foi utilizada uma abordagem binária (ex: detrator vs não detrator).

---

## 🧠 Metodologia Utilizada

O projeto foi estruturado seguindo boas práticas de projetos de Data Science, com separação entre exploração, engenharia de dados e modelagem.

### 1. Análise Exploratória (EDA)

Realizada em notebook, com foco em:

* Entendimento da distribuição dos dados
* Análise da relação entre reclamações e NPS
* Identificação de padrões comportamentais

Principais insights:

* Clientes com reclamações apresentam tendência significativa de menor NPS
* Existe uma separação clara entre grupos com e sem reclamação

---

### 2. Teste Estatístico

Foi aplicado teste de hipótese para validar a diferença entre os grupos:

* Comparação entre clientes com e sem reclamação
* Avaliação da significância estatística da diferença no NPS

Resultado:

* Evidência estatística de que clientes com reclamação possuem menor satisfação

---

### 3. Engenharia de Features

* Criação de variáveis derivadas
* Tratamento de dados
* Preparação para modelagem

---

### 4. Modelagem

Foi utilizado um modelo de:

* **Random Forest Classifier**

Etapas:

* Separação treino/teste
* Treinamento do modelo
* Avaliação com métricas de classificação

Métricas utilizadas:

* Accuracy
* Precision
* Recall
* F1-score
* AUC

---

### 5. Interpretação do Modelo

* Avaliação de importância das variáveis
* Identificação das principais drivers de detratores

Destaque:

* Variável de reclamação aparece como uma das mais relevantes

---

## ⚙️ Como Reproduzir os Resultados

### 1. Clonar o repositório

```bash
git clone <repo_url>
cd <repo>
```

---

### 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

### 4. Estrutura esperada dos dados

Coloque o arquivo de dados em:

```bash
data/raw/desafio_nps_fase_1.csv
```

---

### 5. Executar pipeline

```bash
python -m src.pipeline
```

---

### 6. Exploração adicional

Para análise exploratória:

```bash
notebooks/01_eda_nps.ipynb
```

---

## 📈 Resultados e Insights

* Clientes com reclamação têm maior probabilidade de serem detratores
* A variável de reclamação é altamente relevante para previsão
* Existe evidência estatística que sustenta essa relação
* O modelo preditivo consegue capturar esse comportamento de forma consistente

---

## 💡 Possíveis Evoluções

* Testar novos modelos (Logistic Regression, XGBoost)
* Aplicar técnicas de interpretabilidade (SHAP)
* Implementar pipeline automatizado
* Deploy do modelo como API

---

## 👤 Autor

Gustavo Soares, colocar restante dps ---------------
