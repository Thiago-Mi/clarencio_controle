import control
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize, differential_evolution
import json  
from functools import lru_cache
import pandas as pd

# Configuração dos gráficos
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True

# Seleção do domínio para a função de transferência
domain = st.sidebar.selectbox(
    "Selecione o domínio da função de transferência",
    ["Contínuo (s)", "Discreto (z)"]
)

# Parâmetros específicos para o domínio discreto
if domain == "Discreto (z)":
    Ts = st.sidebar.number_input("Tempo de Amostragem (s)", min_value=0.001, max_value=1.0, value=0.05)
    u_min = st.sidebar.number_input("Limite Mínimo da saída Controle", min_value=-100.0, max_value=100000.0, value=0.0)
    u_max = st.sidebar.number_input("Limite Máximo da saída Controle", min_value=-100.0, max_value=100000.0, value=10.0)
    w_itae = st.sidebar.number_input("Peso ITAE (w_itae)", value=1.0, help="Peso para o critério ITAE")
    w_control_effort = st.sidebar.number_input("Peso Esforço de Controle (w_control_effort)", value=0.1, help="Peso para o critério de Esforço de Controle")
    w_settling_time = st.sidebar.number_input("Peso Tempo de Acomodação (w_settling_time)", value=2.0, help="Peso para o critério de Tempo de Acomodação")
    w_overshoot = st.sidebar.number_input("Peso Sobressinal (w_overshoot)", value=0.5, help="Peso para o critério de Sobressinal")
else:
    Ts = None
    u_min = None
    u_max = None

# Entrada dos coeficientes da função de transferência
if domain == "Contínuo (s)":
    num = st.sidebar.text_input("Numerador (ex: 1 2 3)", "1")
    den = st.sidebar.text_input("Denominador (ex: 1 4 5)", "1 8 15")
else:
    num = st.sidebar.text_input("Numerador em z (coeficientes em ordem decrescente dos poderes de z)", "1 0.5")
    den = st.sidebar.text_input("Denominador em z (coeficientes em ordem decrescente dos poderes de z)", "1 -1.5 0.5")

def create_transfer_function(numerator, denominator, domain, Ts=None):
    """
    Cria a função de transferência de acordo com o domínio selecionado.
    """
    if domain == "Contínuo (s)":
        return control.tf(numerator, denominator)
    else:
        if Ts is None:
            Ts = 0.05
        return control.TransferFunction(numerator, denominator, dt=Ts)

def design_controller(controller_type, params, domain, Ts=None):
    """
    Projetar o controlador com validação e normalização dos coeficientes.
    Agora, para controladores discretos, os termos P, I e D são criados separadamente e somados.
    """
    try:
        # Extrair ganhos e limitar os valores para evitar instabilidade numérica
        Kp = float(params.get('Kp', 0))
        Ki = float(params.get('Ki', 0))
        Kd = float(params.get('Kd', 0))
        max_gain = 1e6
        Kp = np.clip(Kp, -max_gain, max_gain)
        Ki = np.clip(Ki, -max_gain, max_gain)
        Kd = np.clip(Kd, -max_gain, max_gain)
        
        if domain == "Contínuo (s)":
            # Implementação para sistemas contínuos (mantida a mesma)
            controllers = {
                'P': control.TransferFunction([Kp], [1]),
                'I': control.TransferFunction([Ki], [1, 0]),
                'D': control.TransferFunction([Kd * 1, 0], [1]),  # D(s) = Kd * s
                'PI': control.TransferFunction([Kp, Ki], [1, 0]),
                'PD': control.TransferFunction([Kd, Kp], [1, 0]),
                'PID': control.TransferFunction([Kd, Kp, Ki], [1, 0])
            }
            if controller_type.upper() not in controllers:
                raise ValueError(f"Tipo de controlador inválido: {controller_type}")
            return controllers[controller_type.upper()]
        
        else:
            if Ts is None or Ts <= 0:
                raise ValueError("Tempo de amostragem inválido")
            # Implementação para sistemas discretos utilizando Tustin (bilinear)
            # Cada controlador é construído como a soma dos seus blocos.
            if controller_type.upper() == 'P':
                return control.TransferFunction([Kp], [1], dt=Ts)
            elif controller_type.upper() == 'I':
                # I(z) = Ki*Ts/2 * (z+1)/(z-1)
                return control.TransferFunction([Ki*Ts/2, Ki*Ts/2], [1, -1], dt=Ts)
            elif controller_type.upper() == 'D':
                # D(z) = 2*Kd/Ts * (z-1)/(z+1)
                return control.TransferFunction([2*Kd/Ts, -2*Kd/Ts], [1, 1], dt=Ts)
            elif controller_type.upper() == 'PI':
                P = control.TransferFunction([Kp], [1], dt=Ts)
                I = control.TransferFunction([Ki*Ts/2, Ki*Ts/2], [1, -1], dt=Ts)
                return P + I
            elif controller_type.upper() == 'PD':
                P = control.TransferFunction([Kp], [1], dt=Ts)
                D = control.TransferFunction([2*Kd/Ts, -2*Kd/Ts], [1, 1], dt=Ts)
                return P + D
            elif controller_type.upper() == 'PID':                
                P = control.TransferFunction([Kp], [1], dt=Ts)
                I = control.TransferFunction([Ki*Ts/2, Ki*Ts/2], [1, -1], dt=Ts)
                D = control.TransferFunction([2*Kd/Ts, -2*Kd/Ts], [1, 1], dt=Ts)
                return P + I + D
            else:
                raise ValueError(f"Tipo de controlador inválido: {controller_type}")
            
    except Exception as e:
        raise ValueError(f"Erro ao projetar controlador: {str(e)}")


def cost_function(params, G, controller_type, domain, Ts=None, u_min=None, u_max=None, 
                  w_itae=1.0, w_control_effort=0.1, w_settling_time=2.0, w_overshoot=0.5):
    """
    Função de custo para otimização.
    Para o domínio contínuo utiliza ITAE, enquanto que para o discreto também considera o esforço
    de controle, tempo de acomodação e sobressinal, aplicando saturação.
    """
    if domain == "Contínuo (s)":
        Kp, Ki, Kd = params
        C = design_controller(controller_type, {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}, domain)
        L = C * G
        sys_cl = control.feedback(L, 1)
        t, y = control.step_response(sys_cl)
        error = 1 - y
        itae = np.sum(t * np.abs(error))
        return itae
    else:
        Kp, Ki, Kd = params
        C = design_controller(controller_type, {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}, domain, Ts)
        L = C * G
        sys_cl = control.feedback(L, 1)
        t = np.arange(0, 5, Ts)
        t, y = control.step_response(sys_cl, t)
        error = 1 - y
        t_eval, u_temp = control.forced_response(C, T=t, U=error)
        if u_min is not None and u_max is not None:
            u = np.clip(u_temp, u_min, u_max)
        else:
            u = u_temp
        itae = np.sum(t * np.abs(error))
        control_effort = np.sum(np.abs(u))
        steady_state = y[-1]

        # Cálculo do tempo de acomodação
        if abs(steady_state) < 1e-6:
            settling_mask = np.where(np.abs(y) <= 0.02 * np.max(np.abs(y)))[0]
        else:
            settling_mask = np.where(np.abs(y - steady_state) <= 0.02 * abs(steady_state))[0]
        if len(settling_mask) > 0:
            last_violation = 0
            for i in range(1, len(settling_mask)):
                if settling_mask[i] - settling_mask[i-1] > 1:
                    last_violation = settling_mask[i-1]
            settling_time = t[settling_mask[np.where(settling_mask > last_violation)[0][0]]]
        else:
            settling_time = t[-1]
        overshoot = (np.max(y) - steady_state) / abs(steady_state) * 100 if np.max(y) > steady_state else 0
        cost = w_itae * itae + w_control_effort * control_effort + w_settling_time * settling_time + w_overshoot * overshoot
        return cost


def check_stability(sys_obj, domain):
    """
    Verifica a estabilidade do sistema:
      - Contínuo: todos os polos com parte real negativa.
      - Discreto: todos os polos dentro do círculo unitário.
    """
    poles = control.poles(sys_obj)
    if domain == "Contínuo (s)":
        return all(p.real < 0 for p in poles), poles
    else:
        return all(abs(p) < 1 for p in poles), poles

@st.cache_data(hash_funcs={plt.Figure: lambda fig: None})
def plot_response(_sys_obj, title, domain, Ts=None):
    fig, ax = plt.subplots()
    if domain == "Discreto (z)" and Ts is not None:
        t = np.arange(0, 5, Ts)
        t, y = control.step_response(_sys_obj, t)
    else:
        t, y = control.step_response(_sys_obj)
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    return fig, t, y

@st.cache_data(hash_funcs={plt.Figure: lambda fig: None, control.TransferFunction: lambda tf: str(tf)})
def plot_root_locus(sys_obj, title):
    fig = plt.figure()
    try:
        control.root_locus(sys_obj, plot=True)
        plt.title(title, fontsize=14)
        plt.xlabel('Parte Real', fontsize=12)
        plt.ylabel('Parte Imaginária', fontsize=12)
        plt.grid(True)
    except Exception as e:
        plt.close(fig)
        st.error(f"Erro ao plotar lugar das raízes: {e}")
        return None
    return fig


@st.cache_data(hash_funcs={plt.Figure: lambda fig: None, control.TransferFunction: lambda tf: str(tf)})
def plot_bode(sys_obj, title):
    fig = plt.figure()
    try:
        control.bode_plot(sys_obj, dB=True, Hz=True, plot=True)
        plt.title(title)
        plt.grid(True)
    except Exception as e:
        plt.close(fig)
        st.error(f"Erro ao plotar diagrama de Bode: {e}")
        return None
    return fig


@st.cache_data(hash_funcs={plt.Figure: lambda fig: None, control.TransferFunction: lambda tf: str(tf)})
def compare_responses(G, T, title="Comparação das Respostas"):
    fig = plt.figure(figsize=(12, 6))
    T_original = control.feedback(G, 1)
    t_orig, y_orig = control.step_response(T_original)
    plt.plot(t_orig, y_orig, 'r--', label='Sistema Original')
    t_ctrl, y_ctrl = control.step_response(T)
    plt.plot(t_ctrl, y_ctrl, 'b-', label='Sistema Controlado')
    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    return fig


def verify_controller_performance(T, Ts=0.05, domain="Contínuo (s)", u_min=None, u_max=None):
    if domain == "Contínuo (s)":
        t, y = control.step_response(T)
    else:
        t = np.arange(0, 5, Ts)
        t, y = control.step_response(T, t)
        if u_min is not None and u_max is not None:
            y = np.clip(y, u_min, u_max)
    steady_state = y[-1]
    settling_mask = np.where(np.abs(y - steady_state) <= 0.02 * abs(steady_state))[0]
    settling_time = t[settling_mask[0]] if len(settling_mask) > 0 else t[-1]
    overshoot = (np.max(y) - steady_state) / abs(steady_state) * 100 if np.max(y) > steady_state else 0
    stability, _ = check_stability(T, domain)
    return {
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_state': steady_state,
        'stability': stability
    }


def validate_params(Ts=None, u_min=None, u_max=None):
    """
    Garante que o tempo de amostragem seja positivo e que u_min seja menor que u_max.
    """
    if Ts is not None and Ts <= 0:
        raise ValueError("Tempo de amostragem deve ser positivo")
    if u_min is not None and u_max is not None:
        if u_min >= u_max:
            raise ValueError("u_min deve ser menor que u_max")


@st.cache_data(hash_funcs={plt.Figure: lambda fig: None, control.TransferFunction: lambda tf: str(tf)})
def analyze_system(G, domain):
    # Cria a figura do lugar das raízes
    fig_rl = plt.figure(figsize=(12, 6))
    try:
        control.root_locus(G, plot=True)
        plt.title("Lugar das Raízes - Sistema Original")
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid(True)
    except Exception as e:
        plt.close(fig_rl)
        st.error(f"Erro ao plotar lugar das raízes: {e}")
        fig_rl = None

    # Cria a figura do diagrama de Bode
    fig_bode = plt.figure()
    try:
        control.bode_plot(G, dB=True, Hz=True, plot=True)
        plt.title("Diagrama de Bode - Sistema Original")
        plt.grid(True)
    except Exception as e:
        plt.close(fig_bode)
        st.error(f"Erro ao plotar diagrama de Bode: {e}")
        fig_bode = None

    return fig_rl, fig_bode

@st.cache_data(hash_funcs={control.TransferFunction: lambda tf: str(tf)})
def optimize_controller(G, controller_type, domain, Ts=None, u_min=None, u_max=None):
    controller_type = controller_type.upper()
    # Define os bounds e a função de custo de acordo com o controlador selecionado
    if controller_type == 'P':
        bounds = [(0, 10)]
        def cost_function_wrapper(param):
            Kp = param[0]
            params = [Kp, 0, 0]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    elif controller_type == 'I':
        bounds = [(0, 10)]
        def cost_function_wrapper(param):
            Ki = param[0]
            params = [0, Ki, 0]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    elif controller_type == 'D':
        bounds = [(0, 10)]
        def cost_function_wrapper(param):
            Kd = param[0]
            params = [0, 0, Kd]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    elif controller_type == 'PI':
        bounds = [(0, 10), (0, 10)]
        def cost_function_wrapper(param):
            Kp, Ki = param
            params = [Kp, Ki, 0]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    elif controller_type == 'PD':
        bounds = [(0, 10), (0, 10)]
        def cost_function_wrapper(param):
            Kp, Kd = param
            params = [Kp, 0, Kd]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    elif controller_type == 'PID':
        bounds = [(0, 10), (0, 10), (0, 10)]
        def cost_function_wrapper(param):
            Kp, Ki, Kd = param
            params = [Kp, Ki, Kd]
            return cost_function(params, G, controller_type, domain, Ts, u_min, u_max)
    else:
        raise ValueError(f"Tipo de controlador inválido: {controller_type}")
    
    if domain == "Contínuo (s)":
        result = differential_evolution(cost_function_wrapper, bounds=bounds, maxiter=300, popsize=30)
    else:
        def constraint_stability(param):
            try:
                if controller_type == 'P':
                    Kp = param[0]
                    par = [Kp, 0, 0]
                elif controller_type == 'I':
                    Ki = param[0]
                    par = [0, Ki, 0]
                elif controller_type == 'D':
                    Kd = param[0]
                    par = [0, 0, Kd]
                elif controller_type == 'PI':
                    Kp, Ki = param
                    par = [Kp, Ki, 0]
                elif controller_type == 'PD':
                    Kp, Kd = param
                    par = [Kp, 0, Kd]
                elif controller_type == 'PID':
                    par = param
                C = design_controller(controller_type, {'Kp': par[0], 'Ki': par[1], 'Kd': par[2]}, domain, Ts)
                L = C * G
                sys_cl = control.feedback(L, 1)
                poles = control.poles(sys_cl)
                return np.min([1 - abs(p) for p in poles])
            except Exception as e:
                return -1
        def constrained_cost_function(param):
            stability = constraint_stability(param)
            if stability > 0:
                return cost_function_wrapper(param)
            else:
                return float('inf')
        result = differential_evolution(constrained_cost_function, bounds=bounds, maxiter=300, popsize=30)
        
    # Processa o resultado para garantir que sempre sejam retornados 3 parâmetros
    opt = result.x
    if controller_type == 'P':
        params_out = [opt[0], 0, 0]
    elif controller_type == 'I':
        params_out = [0, opt[0], 0]
    elif controller_type == 'D':
        params_out = [0, 0, opt[0]]
    elif controller_type == 'PD':
        params_out = [opt[0], 0, opt[1]]
    elif controller_type == 'PI':
        params_out = [opt[0], opt[1], 0]
    elif controller_type == 'PID':
        params_out = opt.tolist()
    else:
        params_out = opt.tolist()
        
    if result.success:
        return params_out
    else:
        st.warning(f"Otimização não convergiu adequadamente para o controlador {controller_type}")
        return params_out

@st.cache_data
def compare_continuous_controllers(_G):
    controller_types = ['P', 'PI', 'PD', 'PID']
    results = []
    controllers = {}
    for c_type in controller_types:
        try:
            params = optimize_controller(_G, c_type, "Contínuo (s)")
            C = design_controller(c_type, {'Kp': params[0], 'Ki': params[1], 'Kd': params[2]}, "Contínuo (s)")
            T = control.feedback(C * _G, 1)
            performance = verify_controller_performance(T, domain="Contínuo (s)")
            stable, poles = check_stability(T, "Contínuo (s)")
            t, y = control.step_response(T)
            error = 1 - y
            itae = np.sum(t * np.abs(error))
            results.append({
                'Tipo': c_type,
                'Kp': params[0],
                'Ki': params[1],
                'Kd': params[2],
                'Tempo de Acomodação (s)': performance['settling_time'],
                'Sobressinal (%)': performance['overshoot'],
                'Valor Final': performance['steady_state'],
                'ITAE': itae,
                'Estável': stable
            })
            controllers[c_type] = C
        except Exception as e:
            print(f"Erro ao analisar controlador {c_type}: {str(e)}")
            continue
    return pd.DataFrame(results), controllers


@st.cache_data
def compare_discrete_controllers(_G, Ts, u_min, u_max):
    controller_types = ['P', 'PI', 'PD', 'PID']
    results = []
    controllers = {}
    for c_type in controller_types:
        try:
            params = optimize_controller(_G, c_type, "Discreto (z)", Ts, u_min, u_max)
            C = design_controller(c_type, {'Kp': params[0], 'Ki': params[1], 'Kd': params[2]}, "Discreto (z)", Ts)
            T = control.feedback(C * _G, 1)
            performance = verify_controller_performance(T, Ts, "Discreto (z)", u_min, u_max)
            stable, poles = check_stability(T, "Discreto (z)")
            t = np.arange(0, 5, Ts)
            _, y = control.step_response(T, t)
            error = 1 - y
            _, u = control.forced_response(C, T=t, U=error)
            u_sat = np.clip(u, u_min, u_max)
            control_effort = np.sum(np.abs(u_sat))
            results.append({
                'Tipo': c_type,
                'Kp': params[0],
                'Ki': params[1],
                'Kd': params[2],
                'Tempo de Amostragem (s)': Ts,
                'Tempo de Acomodação (s)': performance['settling_time'],
                'Sobressinal (%)': performance['overshoot'],
                'Valor Final': performance['steady_state'],
                'Esforço de Controle': control_effort,
                'Estável': stable
            })
            controllers[c_type] = C
        except Exception as e:
            print(f"Erro ao analisar controlador {c_type}: {str(e)}")
            continue
    return pd.DataFrame(results), controllers


def main():
    st.title("Projeto Integrado de Sistema de Controle")
    st.markdown("""
    ## Interface para Projeto de Controladores
    Esta aplicação permite:
    - Visualizar o comportamento do sistema original
    - Projetar controladores (PID, PI, PD, P, I ou D)
    - Comparar o sistema antes e depois do controle, com análise de desempenho
    """)
    try:
        validate_params(Ts, u_min, u_max)
    except Exception as e:
        st.error(f"Erro na validação dos parâmetros: {e}")
        return

    try:
        numerator = list(map(float, num.split()))
        denominator = list(map(float, den.split()))
        G = create_transfer_function(numerator, denominator, domain, Ts)
        controller_type = st.selectbox("Tipo de Controlador", 
                                       ['PID', 'PI', 'PD', 'P', 'I', 'D'])
    except Exception as e:
        st.error(f"Erro na função de transferência: {e}")
        return

    st.subheader("Análise do Sistema Original")
    try:
        G_closed = control.feedback(G, 1)
        stable_orig, poles_orig = check_stability(G_closed, domain)
        stable_open, poles_open = check_stability(G, domain)
        st.write(f"*Sistema Original Estável (Malha Fechada):* {stable_orig}")
        st.write(f"*Sistema Original Estável (Malha Aberta):* {stable_open}")
        st.write(f"*Polos Originais:* {poles_orig}")
        
        root_locus_fig, bode_fig = analyze_system(G, domain)
        if root_locus_fig:
            st.pyplot(root_locus_fig)
        if bode_fig:
            st.pyplot(bode_fig)
        plot, ___,__ = plot_response(G_closed, "Resposta ao Degrau - Original", domain, Ts)
        st.pyplot(plot)
    except Exception as e:
        st.error(f"Erro na análise do sistema original: {e}")
        return

    st.subheader("Análise do Sistema Controlado")
    
    if st.button("Comparar Todos os Controladores"):
        with st.spinner('Realizando análise comparativa...'):
            try:
                if domain == "Contínuo (s)":
                    results_df, controllers_dict = compare_continuous_controllers(G)
                else:
                    results_df, controllers_dict = compare_discrete_controllers(G, Ts, u_min, u_max)
                
                st.markdown("### Resultados Comparativos")
                st.dataframe(results_df)
                
                plt.figure(figsize=(12, 6))
                # Resposta do sistema original
                _, t_orig, y_orig = plot_response(G_closed, "Resposta ao Degrau - Original", domain, Ts)
                plt.plot(t_orig, y_orig, 'k--', label='Original')

                # Plotar somente os controladores estáveis
                for c_type, C in controllers_dict.items():
                    T = control.feedback(C * G, 1)
                    stable, _ = check_stability(T, domain)
                    if stable:
                        t, y = control.step_response(T)
                        plt.plot(t, y, label=c_type)

                plt.title("Comparação das Respostas ao Degrau")
                plt.xlabel('Tempo')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
                
            except Exception as e:
                st.error(f"Erro na análise comparativa: {e}")
                return

    if st.button("Projetar Controlador"):
        with st.spinner('Otimizando parâmetros...'):
            try:
                if domain == "Contínuo (s)":
                    Kp_opt, Ki_opt, Kd_opt = optimize_controller(G, controller_type, domain)
                else:
                    Kp_opt, Ki_opt, Kd_opt = optimize_controller(G, controller_type, domain, Ts, u_min, u_max)
                
                params = {'Kp': Kp_opt, 'Ki': Ki_opt, 'Kd': Kd_opt}
                C = design_controller(controller_type, params, domain, Ts)
                L = C * G
                T = control.feedback(L, 1)
                stable, poles = check_stability(T, domain)
                stable_controller, poles_controller = check_stability(C, domain)

                st.markdown("### Ganhos do Controlador")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Kp", f"{Kp_opt:.4f}")
                with col2:
                    st.metric("Ki", f"{Ki_opt:.4f}")
                with col3:
                    st.metric("Kd", f"{Kd_opt:.4f}")
                
                st.markdown("### Análise de Estabilidade")
                performance = verify_controller_performance(T, Ts if Ts is not None else 0.05, domain)
                st.write(f"*Sistema Controlado Estável (Malha Fechada):* {stable}")
                st.write(f"*Controlador Estável:* {stable_controller}")
                st.write(f"*Polos do Sistema Controlado:* {poles}")
                
                root_locus_plot = plot_root_locus(L, "Lugar das Raízes - Controlado")
                if root_locus_plot:
                    st.pyplot(root_locus_plot)
                
                bode_plot = plot_bode(L, "Diagrama de Bode - Controlado")
                if bode_plot:
                    st.pyplot(bode_plot)
                
                response_plot, __, __ = plot_response(T, "Resposta ao Degrau - Controlado", domain, Ts)
                if response_plot:
                    st.pyplot(response_plot)
                    
                st.pyplot(compare_responses(G, T, "Comparação das Respostas ao Degrau"))
                
                st.markdown("### Métricas de Desempenho")
                st.metric("Tempo de Acomodação", f"{performance['settling_time']:.3f} s")
                st.metric("Sobressinal", f"{performance['overshoot']:.2f}%")
                st.metric("Valor Final", f"{performance['steady_state']:.4f}")
                
                controller_data = {
                    'controller_type': controller_type,
                    'Kp': Kp_opt,
                    'Ki': Ki_opt,
                    'Kd': Kd_opt
                }
                with open("controller.json", "w") as f:
                    json.dump(controller_data, f, indent=4)
                st.success("Controlador salvo com sucesso em 'controller.json'!")
                
            except Exception as e:
                st.error(f"Erro na otimização: {e}")
                return

if __name__ == "__main__":
    main()
