import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import streamlit as st
import pandas as pd

# ==================================================================
# 1. 기본 상수 및 단위 변환 설정
# ==================================================================
STP_MOLAR_VOLUME = 22414.0  # cm³/mol
BAR_TO_ATM = 0.986923
M3H_TO_CM3S = 1_000_000.0 / 3600.0
M2_TO_CM2 = 10000.0
GPU_TO_STD_UNITS = 1e-6 * 76.0 

# ==================================================================
# 2. MembraneStage 클래스 (물리적 모델 - 기존 코드 그대로 사용)
# ==================================================================
class MembraneStage:
    def __init__(self, name):
        self.name = name
        self.area = 0.0
        self.stage_cut = 0.0
        self.feed_flux = 0.0
        self.feed_comp = None
        self.permeate_flux = 0.0
        self.permeate_comp = None
        self.retentate_flux = 0.0
        self.retentate_comp = None

    def _calc_yi_system(self, x, params):
        n_comp = len(x)
        L, p_u, p_p = params["L"], params["p_u"], params["p_p"]
        x_safe = np.clip(x, 1e-12, 1.0)

        def equations(yi):
            yi_safe = np.clip(yi, 1e-12, 1.0)
            eqs = []
            for i in range(n_comp - 1):
                driving_force_i = L[i] * (p_u * x_safe[i] - p_p * yi_safe[i])
                driving_force_j = L[i + 1] * (p_u * x_safe[i + 1] - p_p * yi_safe[i + 1])
                eqs.append(yi_safe[i] * driving_force_j - yi_safe[i + 1] * driving_force_i)
            eqs.append(np.sum(yi) - 1.0)
            return eqs

        yi_sol, _, ier, _ = fsolve(equations, x_safe.copy(), full_output=True)
        if ier != 1: pass
        return np.clip(yi_sol, 1e-10, 1.0)

    def _odes(self, A, y_state, params):
        n_comp = len(params["L"])
        x = y_state[:n_comp]
        Lu = y_state[n_comp]

        if Lu < 1e-9:
            return np.zeros(n_comp + 1)

        x = np.maximum(x, 0)
        sum_x = np.sum(x)
        if sum_x > 0:
            x /= sum_x

        yi = self._calc_yi_system(x, params)
        Ji = params["L"] * (params["p_u"] * x - params["p_p"] * yi)
        Ji = np.maximum(Ji, 0) 
        
        dLu_dA = -np.sum(Ji)
        dxi_dA = (x * np.sum(Ji) - Ji) / Lu

        return np.hstack((dxi_dA, dLu_dA))

    def run(self, feed_flux, feed_comp, area_target, params):
        if not np.isclose(np.sum(feed_comp), 1.0):
             feed_comp = feed_comp / np.sum(feed_comp)
        
        self.feed_flux = feed_flux
        self.feed_comp = feed_comp
        n_comp = len(feed_comp)
        y_state0 = np.hstack((feed_comp, feed_flux))

        def retentate_empty(t, y, params): 
            return y[-1] - (feed_flux * 0.001)
            
        retentate_empty.terminal = True
        retentate_empty.direction = -1

        sol = solve_ivp(
            fun=self._odes,
            t_span=[0, area_target],
            y0=y_state0,
            method='RK45',
            args=(params,),
            events=retentate_empty
        )

        self.area = sol.t[-1]
        final_y_state = sol.y[:, -1]

        self.retentate_flux = max(final_y_state[n_comp], 0.0)
        self.retentate_comp = np.maximum(final_y_state[:n_comp], 0)
        if np.sum(self.retentate_comp) > 0:
            self.retentate_comp /= np.sum(self.retentate_comp)

        self.permeate_flux = self.feed_flux - self.retentate_flux
        
        if self.permeate_flux > 1e-9:
            feed_moles = self.feed_flux * self.feed_comp
            ret_moles = self.retentate_flux * self.retentate_comp
            permeate_moles = np.maximum(feed_moles - ret_moles, 0)
            
            if np.sum(permeate_moles) > 0:
                self.permeate_comp = permeate_moles / np.sum(permeate_moles)
            else:
                self.permeate_comp = np.zeros(n_comp)
        else:
            self.permeate_comp = np.zeros(n_comp)
            
        self.stage_cut = self.permeate_flux / self.feed_flux if self.feed_flux > 1e-9 else 0.0
        return True


# ==================================================================
# 3. Streamlit UI (1단 공정용으로 단순화)
# ==================================================================
st.set_page_config(page_title="1-Stage Membrane Sim", layout="wide")
st.title("🧪 1-Stage (N2/O2) Membrane Simulator")
st.markdown("교차흐름(Cross-flow) 모델을 적용한 단일 분리막 공정 시뮬레이터입니다.")

with st.sidebar:
    st.header("⚙️ 1. 원료 (Feed) 조건")
    feed_flux_m3h = st.number_input("Feed 총 유량 (m³/h)", value=100.0)
    col1, col2 = st.columns(2)
    with col1: comp_n2 = st.number_input("N2 몰분율", value=0.79, format="%.2f")
    with col2: comp_o2 = st.number_input("O2 몰분율", value=0.21, format="%.2f")

    st.header("⚙️ 2. 분리막 및 운전 조건")
    area_m2 = st.number_input("분리막 면적 (m²)", value=80.0)
    pu = st.number_input("주입부(Upstream) 압력 (bar)", value=12.0)
    pp = st.number_input("투과부(Permeate) 압력 (bar)", value=1.0)
    
    st.subheader("Permeance (GPU)")
    l_n2 = st.number_input("N2 GPU", value=50.0)
    l_o2 = st.number_input("O2 GPU", value=500.0)

    st.write("---")
    btn_run = st.button("시뮬레이션 실행 (Run)", type="primary", use_container_width=True)


if btn_run:
    with st.spinner("미분방정식(ODE) 계산 중..."):
        # 1. 입력값 정리 및 단위 변환
        raw_feed_comp = np.array([comp_n2, comp_o2])
        if abs(sum(raw_feed_comp) - 1.0) > 1e-3:
            raw_feed_comp = raw_feed_comp / sum(raw_feed_comp)

        # m3/h -> mol/s 변환
        feed_flux_mol = (feed_flux_m3h * M3H_TO_CM3S) / STP_MOLAR_VOLUME
        
        # GPU -> 단위 변환
        gpu_arr = np.array([l_n2, l_o2])
        L_std = gpu_arr * GPU_TO_STD_UNITS
        L_mol = L_std / STP_MOLAR_VOLUME
        params = {"L": L_mol, "p_u": pu * BAR_TO_ATM, "p_p": pp * BAR_TO_ATM}
        
        area_cm2 = area_m2 * M2_TO_CM2

        # 2. 계산 실행
        stage = MembraneStage("1-Stage")
        success = stage.run(feed_flux_mol, raw_feed_comp, area_cm2, params)

    if success:
        st.success("✅ 계산이 성공적으로 완료되었습니다!")
        
        vol_conv = STP_MOLAR_VOLUME * (3600/1e6) # mol/s -> m3/h 변환 상수
        
        # 3. 핵심 결과 대시보드
        c1, c2, c3 = st.columns(3)
        c1.metric("투과부(Permeate) O2 순도", f"{stage.permeate_comp[1]*100:.2f} %")
        c2.metric("투과부(Permeate) 유량", f"{stage.permeate_flux * vol_conv:.2f} m³/h")
        c3.metric("Stage Cut (회수율)", f"{stage.stage_cut:.3f}")

        if stage.stage_cut > 0.95:
            st.error("⚠️ 경고: Stage Cut이 1.0에 가깝습니다. 면적(Area)이 너무 커서 사실상 모든 기체가 투과되었습니다.")

        # 4. 상세 결과 표 출력
        st.subheader("📊 흐름별 상세 데이터")
        
        result_data = {
            "Stream": ["Feed (주입부)", "Permeate (투과부)", "Retentate (잔류부)"],
            "Flow Rate (m³/h)": [
                stage.feed_flux * vol_conv, 
                stage.permeate_flux * vol_conv, 
                stage.retentate_flux * vol_conv
            ],
            "O2 Concentration (%)": [
                stage.feed_comp[1] * 100,
                stage.permeate_comp[1] * 100,
                stage.retentate_comp[1] * 100
            ],
            "N2 Concentration (%)": [
                stage.feed_comp[0] * 100,
                stage.permeate_comp[0] * 100,
                stage.retentate_comp[0] * 100
            ]
        }
        
        df = pd.DataFrame(result_data)
        df.set_index("Stream", inplace=True)
        
        # O2 순도가 94% 이상이면 투과부 초록색 하이라이트
        def highlight_row(row):
            if row.name == "Permeate (투과부)" and row["O2 Concentration (%)"] > 94.0:
                return ['background-color: #d4edda; color: green; font-weight: bold'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.style.format("{:.2f}").apply(highlight_row, axis=1), use_container_width=True)