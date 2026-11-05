# Previsão de churn com aprendizado de máquina

Projeto da segunda parte da disciplina Sistemas Inteligentes (CIn/UFPE, 2026.2). A pergunta que a gente tenta responder: dado um cliente de uma operadora, ele vai cancelar o plano nos próximos 60 dias?

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

## Equipe

| Integrante | Contato |
|---|---|
| Henrique Amorim | [hlpaa@cin.ufpe.br](mailto:hlpaa@cin.ufpe.br) |
| Henrique Passos | [hap@cin.ufpe.br](mailto:hap@cin.ufpe.br) |
| José Francisco Cruz Neto | [jfcn3@cin.ufpe.br](mailto:jfcn3@cin.ufpe.br) |
| Karoline Viana | [kvf@cin.ufpe.br](mailto:kvf@cin.ufpe.br) |
| Manoel Villarim | [mfav@cin.ufpe.br](mailto:mfav@cin.ufpe.br) |

## O problema

A operadora tem 5.000 clientes e quer saber quais deles vão sair antes que isso aconteça, para poder agir a tempo. Reter um cliente costuma sair mais barato que conquistar outro.

Do ponto de vista técnico é uma classificação binária: `churn = 1` se o cliente cancela em até 60 dias e `churn = 0` se fica.

## Miniprojetos

| # | Miniprojeto | Status |
|---|-------------|--------|
| 1 | Análise, tratamento e construção da base de dados de churn | Concluído |
| 2 | Algoritmos supervisionados (Árvores de Decisão, Gradient Boosting, MLP) | A fazer |
| 3 | Algoritmos não supervisionados (K-means, K-prototypes, DBSCAN) | A fazer |

## Miniprojeto 1

Todo o trabalho está no notebook [`Analise_tratamento_Churn.ipynb`](Analise_tratamento_Churn.ipynb). Os passos, na ordem em que aparecem:

```mermaid
flowchart LR
    A[Base bruta<br/>5.020 linhas] --> B[Remover<br/>duplicados]
    B --> C[Visão de dados<br/>+ data de corte]
    C --> D[Divisão<br/>70% treino / 30% teste]
    D --> E[Qualidade<br/>4 dimensões]
    E --> F[Faltantes<br/>moda + KNN]
    F --> G[Exploração<br/>e transformação]
    G --> H[Base transformada]
```

### Base bruta

A base veio com valores faltantes, mensalidades negativas ou absurdamente altas, textos escritos de formas diferentes (`bienal `, `Cartao`) e 20 clientes duplicados.

<p align="center"><img src="imagens/faltantes_base_bruta.png" width="90%" alt="Valores faltantes na base bruta"></p>

### Visão de dados e data de corte

Montamos uma linha por cliente. A data de corte fica 60 dias antes do fim da base: os atributos usam só o que se sabia até ali, e o churn é observado nos 60 dias seguintes. Calcular o tempo de casa com a data do cancelamento seria dar o alvo de presente ao modelo, então evitamos isso.

### Divisão treino e teste

Separamos 70% para treino e 30% para teste, estratificando pelo churn, logo depois de montar a visão de dados e antes de calcular qualquer estatística. Moda, KNN, média e desvio padrão saem só do treino. O teste só recebe a transformação (`.fit` no treino, `.transform` no teste).

<p align="center"><img src="imagens/estratificacao.png" width="85%" alt="Estratificação do churn em treino e teste"></p>

### Qualidade dos dados

Medimos completude, validade, consistência e unicidade antes e depois do tratamento. Os valores fora das faixas do inventário viraram nulos, e a validade foi a 100%. A completude que sobrou é resolvida na imputação.

<p align="center">
  <img src="imagens/qualidade_antes.png" width="48%" alt="Qualidade antes do tratamento">
  <img src="imagens/qualidade_depois.png" width="48%" alt="Qualidade depois do tratamento">
</p>

### Faltantes

Para cada coluna analisamos se a ausência parecia aleatória (MCAR) ou ligada a outras variáveis (MAR). As categóricas foram preenchidas com a moda do treino e as numéricas com `KNNImputer`.

### Exploração e transformação

Olhamos a distribuição de cada variável por classe, a estatística KS e a correlação de Pearson. Depois discretizamos as numéricas em faixas escolhidas por uma árvore rasa (bins preditivos) e padronizamos com `StandardScaler`.

## Números

| | |
|---|---|
| Registros brutos | 5.020 (20 duplicados removidos, sobram 5.000 clientes) |
| Clientes elegíveis na data de corte | 4.990 |
| Churn na janela de 60 dias | cerca de 35,5% |
| Treino e teste | 3.493 e 1.497, com a mesma proporção de churn nos dois |

## O que os dados mostram

O tempo de casa é o que mais protege. O churn passa de 60% nos primeiros 6 meses e cai para menos de 10% depois de 5 anos. Os chamados de suporte puxam para o outro lado: a taxa vai de uns 22% em quem não abriu nenhum para mais de 60% em quem abriu 5 ou mais.

<p align="center"><img src="imagens/churn_tempo_chamados.png" width="95%" alt="Taxa de churn por tempo de cliente e por quantidade de chamados"></p>

O contrato mensal tem cerca de 48% de churn, contra uns 24% no anual e 15% no bienal. A forma de pagamento quase não muda o resultado.

<p align="center"><img src="imagens/contrato_pagamento.png" width="55%" alt="Churn por tipo de contrato e forma de pagamento"></p>

Juntando três fatores (até 18 meses de casa, 2 ou mais chamados e pelo menos 1 atraso), criamos o "combo de risco triplo". Ele pega cerca de 15% dos clientes, e esses têm uns 63% de churn, o dobro do resto.

<p align="center"><img src="imagens/combo_risco_triplo.png" width="85%" alt="Taxa de churn do combo de risco triplo"></p>

As variáveis têm pouca correlação entre si, então nenhuma é redundante.

<p align="center"><img src="imagens/correlacao.png" width="55%" alt="Matriz de correlação de Pearson"></p>

A base tratada tem sinal suficiente para o próximo miniprojeto. Tempo de cliente, chamados, atrasos e tipo de contrato são as variáveis que mais separam as duas classes.

## Estrutura do repositório

```
churn-previsao-ml/
├── Analise_tratamento_Churn.ipynb   # análise e tratamento (miniprojeto 1)
├── base_churn_5000_VF2_aluno.xlsx   # base bruta
├── treino_tratado.csv               # base transformada, treino (70%)
├── teste_tratado.csv                # base transformada, teste (30%)
├── imagens/                         # gráficos usados neste README
└── README.md
```

## Como rodar

```bash
git clone https://github.com/Henrique-apassos/churn-previsao-ml.git
cd churn-previsao-ml
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl jupyter
jupyter notebook Analise_tratamento_Churn.ipynb
```

Também dá para abrir no Google Colab, enviando a base `.xlsx` junto com o notebook. Ao rodar tudo até o fim, ele gera `treino_tratado.csv` e `teste_tratado.csv`.

Bibliotecas usadas: pandas, NumPy, scikit-learn, SciPy, Matplotlib e Seaborn.
