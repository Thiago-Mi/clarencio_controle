# Projeto Integrado de Controle de Sistemas

Este projeto implementa uma aplicação interativa utilizando **Streamlit** para projetar, otimizar e comparar controladores de sistemas no domínio contínuo e discreto.

## Funcionalidades

- Seleção do domínio da função de transferência (**Contínuo** ou **Discreto**);
- Entrada de coeficientes da função de transferência;
- Definição de parâmetros como tempo de amostragem e limites de saturação do sinal de controle (para sistemas discretos);
- Projeto de controladores **P, I, D, PI, PD, PID**;
- Discretização utilizando a transformação **bilinear/Tustin** para sistemas discretos;
- Otimização dos parâmetros do controlador usando **evolução diferencial**;
- Visualização gráfica dos resultados (**resposta ao degrau, lugar das raízes, diagrama de Bode**);
- Comparação de controladores com métricas de desempenho.

## Tecnologias Utilizadas

- **Python**
- **Streamlit** (Interface gráfica)
- **Control** (Manipulação de funções de transferência e análise de sistemas)
- **NumPy** (Operações numéricas)
- **Matplotlib** (Visualização de gráficos)
- **SciPy** (Otimização dos ganhos do controlador)
- **Pandas** (Exibição de resultados comparativos)

## Estrutura do Código

O código é modular e dividido em funções responsáveis por diferentes tarefas:

### 1. Criação da Função de Transferência

```python
create_transfer_function(numerator, denominator, domain, Ts=None)
```

Cria a função de transferência de acordo com o domínio escolhido.

### 2. Projeto do Controlador

```python
design_controller(controller_type, params, domain, Ts=None)
```

Projeta controladores **P, I, D, PI, PD e PID** para sistemas contínuos e discretos, utilizando a discretização bilinear (Tustin) quando necessário.

### 3. Otimização dos Parâmetros do Controlador

```python
optimize_controller(G, controller_type, domain, Ts=None, u_min=None, u_max=None)
```

Otimiza os ganhos **Kp, Ki e Kd** para minimizar a função de custo baseada no **ITAE**, esforço de controle, tempo de acomodação e sobressinal.

### 4. Análise de Estabilidade

```python
check_stability(sys_obj, domain)
```

Verifica se o sistema controlado é estável.

### 5. Visualização dos Resultados

```python
plot_response(sys_obj, title, domain, Ts=None)
plot_root_locus(sys_obj, title)
plot_bode(sys_obj, title)
compare_responses(G, T, title)
```

Gera gráficos para análise do sistema antes e depois do controle.

### 6. Comparação de Controladores

```python
compare_continuous_controllers(G)
compare_discrete_controllers(G, Ts, u_min, u_max)
```

Compara diferentes tipos de controladores e exibe métricas como **tempo de acomodação, sobressinal e ITAE**.

## Como Executar

1. Instale as dependências:
   ```sh
   pip install streamlit control numpy matplotlib scipy pandas
   ```
2. Execute o aplicativo:
   ```sh
   streamlit run prototipo_4.py
   ```
   
## Autor

Desenvolvido por [Thiago Miranda Aboboreira].

