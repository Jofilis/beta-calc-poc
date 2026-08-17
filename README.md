# BetaCalculo

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen)
![Domain](https://img.shields.io/badge/domain-Quantitative%20Finance-orange)

O **BetaCalculo** é uma aplicação em Python para análise quantitativa de risco sistemático que estima a sensibilidade dos retornos de um ativo em relação a um benchmark de mercado. O projeto transforma séries históricas de preços em análises econométricas interpretáveis, avaliando o comportamento do Beta sob diferentes frequências de amostragem (diária, semanal e mensal) e janelas temporais.

---

## 📌 Principais Funcionalidades

- **Dupla Metodologia de Cálculo:** Validação do Beta via fórmula analítica ($\text{Cov}/\text{Var}$) e via Regressão Linear OLS (`statsmodels`).
- **Análise Multi-Frequência:** Recálculo e comparação do Beta em frequências Diária, Semanal (`resample W-FRI`) e Mensal (`resample ME`).
- **Análise Multi-Período:** Estudo de estabilidade temporal em janelas dinâmicas (ex: 1, 3 e 5 anos).
- **Indicadores Complementares:** Estimação de Alpha do modelo linear, Coeficiente de Determinação ($R^2$) e contagem do tamanho amostral.
- **Processamento em Lote (Batch):** Análise automatizada para constituintes do Ibovespa (`^BVSP`) e S&P 500 (`^GSPC`).
- **Visualização Quantitativa:** Scatter plots com reta OLS e gráficos comparativos de estabilidade do Beta.
- **Relatórios Econômicos Automatizados:** Interpretação textual contextualizada da magnitude das métricas obtidas.

---

## 📐 Fundamentação Econométrica

O núcleo do projeto baseia-se na relação de regressão linear simples:

$$R_i = \alpha + \beta R_m + \epsilon$$

Onde:
- $R_i$: Retorno do ativo ($R_i = \frac{P_t}{P_{t-1}} - 1$)
- $R_m$: Retorno do benchmark (mercado)
- $\alpha$: Intercepto da regressão (retorno médio não explicado pelo mercado)
- $\beta$: Coeficiente angular (risco sistemático / sensibilidade ao mercado)
- $\epsilon$: Termo de erro idiossincrático

### Equivalência Matemática
O projeto garante a consistência matemática do cálculo avaliando o Beta por dois caminhos distintos:

$$\beta = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}$$

> **Nota Econométrica sobre o Alpha:** O Alpha reportado é o intercepto do modelo OLS linear de retornos simples. Ele representa o retorno residual do ativo sob o modelo especificado e não deve ser confundido diretamente com o Alpha de Jensen (que exige ajuste por taxa livre de risco $R_f$).

---

## 🔄 Fluxo de Dados e Arquitetura

O alinhamento entre ativo e mercado é realizado por um `inner join` temporal rigoroso no índice de datas, garantindo que retornos sejam comparados estritamente no mesmo pregão.

```text
                  Yahoo Finance (yfinance)
                             │ (auto_adjust=True)
                             ▼
                    Preços Históricos
                             │
                             ▼
                    calculate_returns()
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           Diário         Semanal         Mensal
       (observações)  (W-FRI .last())  (ME .last())
              └──────────────┬──────────────┘
                             ▼
               Inner Join Temporal por Data
                             │
                             ▼
                 Validação Matemática
            (Len > 2, Var(Rm) > 0, Sem NaN)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   Cálculo Cov/Var vs. OLS             Métricas Estatísticas
   (Consistência do Beta)              (Alpha, R², Obs)
              │                             │
              └──────────────┬──────────────┘
                             ▼
                   Gráficos e Relatórios