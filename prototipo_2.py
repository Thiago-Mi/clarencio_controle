import control
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
import pickle
from control import c2d, TransferFunction


# Configuração inicial do matplotlib
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True

def create_transfer_function(numerator, denominator):
    """
    Cria um objeto de função de transferência diretamente no domínio Z
    
    Args:
        numerator: Coeficientes do numerador em Z
        denominator: Coeficientes do denominador em Z
    Returns:
        TransferFunction: Função de transferência em Z
    """
    return control.TransferFunction(numerator, denominator, dt=True)  # dt=True indica sistema discreto


def design_controller(controller_type, params, Ts=0.05):
    """
    Projeta o controlador especificado no domínio Z usando transformação bilinear
    
    Args:
        controller_type: Tipo do controlador ('P', 'I', 'D', 'PI', 'PD', 'PID')
        params: Dicionário com os parâmetros Kp, Ki, Kd
        Ts: Período de amostragem
    Returns:
        TransferFunction: Controlador discreto
    """
    z = TransferFunction.z
    Kp = params.get('Kp', 0)
    Ki = params.get('Ki', 0)
    Kd = params.get('Kd', 0)

    # Transformação bilinear (Tustin)
    # s = 2/Ts * (z-1)/(z+1)
    controllers = {
        'P': Kp,
        'I': Ki * Ts/2 * (z+1)/(z-1),  # Integrador bilinear
        'D': Kd * 2/Ts * (z-1)/(z+1),  # Derivador bilinear
        'PI': Kp + Ki * Ts/2 * (z+1)/(z-1),
        'PD': Kp + Kd * 2/Ts * (z-1)/(z+1),
        'PID': (Kp + Ki * Ts/2 * (z+1)/(z-1) + Kd * 2/Ts * (z-1)/(z+1))
    }
    
    if controller_type not in controllers:
        raise ValueError("Tipo de controlador inválido.")
    
    return controllers[controller_type.upper()]
def saturate_output(value, u_min=-10, u_max=10):
    """Limita a saída do controlador"""
    return np.clip(value, u_min, u_max)

def cost_function(params, G, controller_type, Ts=0.05, u_min=-10, u_max=10):
    """Função de custo para otimização com saturação e tempo de acomodação"""
    Kp, Ki, Kd = params
    
    # Criar controlador discreto
    C = design_controller(controller_type, {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}, Ts)
    
    # Sistema em malha fechada
    L = C * G
    sys_cl = control.feedback(L, 1)
    
    # Simulação
    t = np.arange(0, 5, Ts)  # 5 segundos de simulação
    t, y = control.step_response(sys_cl, t)
    
    # Calcular sinal de controle
    error = 1 - y
    
    # Implementar saturação no sinal de controle
    t_eval, u_temp = control.forced_response(C, T=t, U=error)
    u = np.clip(u_temp, u_min, u_max)
    
    # Critérios de desempenho
    itae = np.sum(t * np.abs(error))  # Erro ponderado no tempo
    control_effort = np.sum(np.abs(u))  # Esforço de controle
    
    # Cálculo do tempo de acomodação (2%)
    steady_state = y[-1]
    settling_mask = np.where(np.abs(y - steady_state) <= 0.02 * steady_state)[0]
    settling_time = t[settling_mask[0]] if len(settling_mask) > 0 else t[-1]
    
    # Cálculo do sobressinal
    overshoot = (np.max(y) - steady_state) / steady_state * 100 if np.max(y) > steady_state else 0
    
    # Pesos para cada critério
    w1 = 1.0    # Peso para ITAE
    w2 = 0.1    # Peso para esforço de controle
    w3 = 2.0    # Peso para tempo de acomodação
    w4 = 0.5    # Peso para sobressinal
    
    # Função de custo combinada
    cost = (w1 * itae + 
            w2 * control_effort + 
            w3 * settling_time + 
            w4 * overshoot)
    
    return cost

def optimize_controller(G, controller_type, u_min=-10, u_max=10, Ts=0.05):
    """Otimiza os parâmetros do controlador com múltiplos critérios"""
    
    def constraint_stability(params):
        """Restrição para garantir estabilidade"""
        try:
            Kp, Ki, Kd = params
            C = design_controller(controller_type, {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}, Ts)
            L = C * G
            sys_cl = control.feedback(L, 1)
            poles = control.poles(sys_cl)
            return np.min([1 - abs(p) for p in poles])  # Positivo se estável
        except:
            return -1
    
    # Ajuste dos limites baseado no tipo de controlador
    if controller_type == 'P':
        bounds = [(0, 50), (0, 0), (0, 0)]
    elif controller_type == 'PI':
        bounds = [(0, 50), (0, 50), (0, 0)]
    elif controller_type == 'PD':
        bounds = [(0, 50), (0, 0), (0, 50)]
    else:  # PID
        bounds = [(0, 50), (0, 50), (0, 50)]

    constraints = {'type': 'ineq', 'fun': constraint_stability}
    
    try:
        # Múltiplas tentativas com diferentes pontos iniciais
        best_result = None
        best_cost = float('inf')
        
        initial_guesses = [
            [1.0, 0.5, 0.2],  # Conservador
            [5.0, 2.0, 1.0],  # Moderado
            [10.0, 5.0, 2.0]  # Agressivo
        ]
        
        for x0 in initial_guesses:
            result = minimize(
                lambda p: cost_function(p, G, controller_type, Ts, u_min, u_max),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success and result.fun < best_cost:
                best_result = result
                best_cost = result.fun
        
        if best_result is None:
            st.warning("Nenhuma otimização convergiu adequadamente")
            return [0, 0, 0]
            
        return best_result.x
        
    except Exception as e:
        st.error(f"Erro na otimização: {str(e)}")
        return [0, 0, 0]
def check_stability(sys):
    """
    Verifica a estabilidade do sistema discreto
    Retorna True se todos os polos estiverem dentro do círculo unitário
    """
    poles = control.poles(sys)
    return all(abs(p) < 1 for p in poles), poles

def plot_response(sys, title):
    """Plota a resposta ao degrau"""
    plt.figure()
    t, y = control.step_response(sys)
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    return plt

def plot_root_locus(sys, title):
    """Plota o lugar das raízes"""
    # plt.figure(figsize=(15, 10))  # Aumentado o tamanho da figura
    control.root_locus(sys, plot=True)
    plt.title(title, fontsize=14)
    plt.xlabel('Parte Real', fontsize=12)
    plt.ylabel('Parte Imaginária', fontsize=12)
    plt.grid(True)
    return plt

def plot_bode(sys, title):
    """Plota o diagrama de Bode"""
    plt.figure()
    control.bode_plot(sys, plot=True)
    plt.title(title)
    plt.grid(True)
    return plt

def compare_responses(G, T, title="Comparação das Respostas"):
    """
    Compara as respostas do sistema original e controlado
    
    Args:
        G (TransferFunction): Sistema original
        T (TransferFunction): Sistema controlado
    """
    plt.figure(figsize=(12, 6))
    
    # Resposta do sistema original
    t_orig, y_orig = control.step_response(control.feedback(G, 1))
    plt.plot(t_orig, y_orig, 'r--', label='Sistema Original')
    
    # Resposta do sistema controlado
    t_ctrl, y_ctrl = control.step_response(T)
    plt.plot(t_ctrl, y_ctrl, 'b-', label='Sistema Controlado')
    
    plt.title(title)
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    return plt

def analyze_system(G):
    """
    Analisa o sistema original
    
    Args:
        G (TransferFunction): Função de transferência da planta
    """
    # Lugar das raízes
    plt.figure(figsize=(8, 6))
    control.root_locus(G, plot=True)
    plt.title("Lugar das Raízes - Sistema Original")
    plt.grid(True)
    root_locus_plot = plt
    
    root_locus_plot = plot_root_locus(G,"Lugar das Raizes - Sistema Original")
    
    # Diagrama de Bode
    plt.figure(figsize=(10, 8))
    control.bode_plot(G, plot=True)
    plt.title("Diagrama de Bode - Sistema Original")
    plt.grid(True)
    bode_plot = plt
    
    return root_locus_plot, bode_plot

def verify_controller_performance(T, Ts=0.05, settling_threshold=0.02):
    """
    Verifica o desempenho do controlador projetado
    
    Args:
        T: Sistema em malha fechada
        Ts: Tempo de amostragem
        settling_threshold: Critério para tempo de acomodação (2% padrão)
    """
    t = np.arange(0, 5, Ts)
    t, y = control.step_response(T, t)
    
    # Métricas de desempenho
    steady_state = y[-1]
    settling_mask = np.where(np.abs(y - steady_state) <= settling_threshold * steady_state)[0]
    settling_time = t[settling_mask[0]] if len(settling_mask) > 0 else t[-1]
    overshoot = (np.max(y) - steady_state) / steady_state * 100 if np.max(y) > steady_state else 0
    
    performance = {
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_state': steady_state,
        'stability': all(abs(p) < 1 for p in control.poles(T))
    }
    
    return performance

def main():
    """Interface principal usando Streamlit"""
    st.set_page_config(layout="wide")
    
    st.title("Sistema de Controle Digital")
    st.markdown("""
    ## Projeto de Controladores Discretos
    Esta aplicação trabalha com funções de transferência no domínio Z.
    
    ### Como inserir a função de transferência:
    Para G(z) = (z + 0.5)/(z² - 1.5z + 0.5), digite:
    - Numerador: 1 0.5
    - Denominador: 1 -1.5 0.5
    
    Os coeficientes devem ser inseridos em ordem decrescente das potências de z.
    """)

    
    
    # Entrada da função de transferência
    st.sidebar.header("Função de Transferência G(z)")
    num = st.sidebar.text_input("Numerador em z (ex: 1 0.5)", "0.00128 0.004468 0.001022")
    den = st.sidebar.text_input("Denominador em z (ex: 1 -1.5 0.5)", "1 -2.192 1.881 -0.6382")
    
    # Parâmetros de saturação
    st.sidebar.header("Parâmetros de Saturação")
    u_min = st.sidebar.number_input("Limite Mínimo do Controle", -100.0, 0.0, -10.0)
    u_max = st.sidebar.number_input("Limite Máximo do Controle", 0.0, 100.0, 10.0)
    
    # Tempo de amostragem apenas para visualização
    Ts = st.sidebar.number_input("Tempo de Amostragem (s)", 0.001, 1.0, 0.05)
    
    
    # Seleção do controlador
    controller_type = st.sidebar.selectbox(
        "Tipo de Controlador",
        ['PID', 'PI', 'PD', 'P', 'I', 'D']
    )

    # Conversão dos coeficientes para lista de floats
    try:
        numerator = list(map(float, num.split()))
        denominator = list(map(float, den.split()))
        G = create_transfer_function(numerator, denominator)
    except Exception as e:
        st.error(f"Erro na função de transferência: {e}")
        return


    # Análise do sistema original
    st.subheader("Análise do Sistema Original")
    try:
        # Sistema em malha fechada original
        G_closed = control.feedback(G, 1)
        stable_orig, poles_orig = check_stability(G_closed)
        
        st.write(f"*Sistema Original Estável:* {stable_orig}")
        st.write(f"*Polos Originais:* {poles_orig}")

        # Gráficos do sistema original
        root_locus_orig, bode_orig = analyze_system(G)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(root_locus_orig)
        with col2:
            st.pyplot(bode_orig)
            
        st.pyplot(plot_response(G_closed, "Resposta ao Degrau - Sistema Original"))
        
    except Exception as e:
        st.error(f"Erro na análise do sistema original: {e}")
        return
    
    # Otimização dos parâmetros
    # Otimização dos parâmetros
    if st.sidebar.button("Projetar Controlador"):
        with st.spinner('Otimizando parâmetros...'):
            Kp_opt, Ki_opt, Kd_opt = optimize_controller(G, controller_type, u_min, u_max, Ts)
            params = {'Kp': Kp_opt, 'Ki': Ki_opt, 'Kd': Kd_opt}
            
            # Criação do controlador
            C = design_controller(controller_type, params)
            
            # Exibição dos parâmetros otimizados
            st.subheader("Parâmetros do Controlador Otimizado")
            st.write(f"**Tipo de Controlador:** {controller_type}")
            st.write(f"**Kp:** {Kp_opt:.4f}")
            st.write(f"**Ki:** {Ki_opt:.4f}")
            st.write(f"**Kd:** {Kd_opt:.4f}")
            
            # Exibição da função de transferência do controlador
            st.write("**Função de Transferência do Controlador:**")
            st.latex(f"C(z) = {C}")
            
            L = C * G
            T = control.feedback(L, 1)
            
            # Verificação de estabilidade
            stable, poles = check_stability(T)
            
            # Verificação de desempenho
            performance = verify_controller_performance(T, Ts)
            
            st.subheader("Análise de Desempenho do Controlador")
            st.write(f"**Tempo de Acomodação:** {performance['settling_time']:.3f} s")
            st.write(f"**Sobressinal:** {performance['overshoot']:.2f}%")
            st.write(f"**Valor Final:** {performance['steady_state']:.4f}")
            st.write(f"**Sistema Estável:** {performance['stability']}")
            
             # Adicione a comparação das respostas
            st.subheader("Comparação Antes e Depois do Controle")
            st.pyplot(compare_responses(G, T, "Comparação das Respostas ao Degrau"))
            
            # Mostre os gráficos em colunas
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_root_locus(L, "Lugar das Raízes - Sistema Controlado"))
            with col2:
                st.pyplot(plot_bode(L, "Diagrama de Bode - Sistema Controlado"))

            # # Adicione métricas de desempenho
            # t, y = control.step_response(T)
            # settling_time = np.where(np.abs(y - y[-1]) <= 0.02 * y[-1])[0][0] * (t[1] - t[0])
            # overshoot = (np.max(y) - y[-1]) / y[-1] * 100 if np.max(y) > y[-1] else 0
            
            # st.subheader("Métricas de Desempenho")
            # st.write(f"*Tempo de Acomodação (2%):* {settling_time:.2f} s")
            # st.write(f"*Sobressinal:* {overshoot:.2f}%")
            
            # Salvar resultados
            with open("controlador.pkl", "wb") as f:
                pickle.dump((controller_type, C), f)
            st.success("Controlador salvo com sucesso!")

if __name__ == "__main__":
    main()