"""
OTU (Ocean Tuning Unit) — Deep-Sea Forced Aeration Model
=========================================================
目的:
  チリ沖・赤道域スーパーエルニーニョ熱暴走時に生じる海洋炭素吸収源喪失
  （推定 10 %）を抑制するため、深海エアレーション強制駆動によって
  表層温度・CO2 分圧・生物ポンプ効率を正常範囲に回復させる
  物理化学シミュレーションのベースロジック。

物理層:
  1. 海洋熱コンテンツ（OHC）2 層モデル（表層 / 深層）
  2. 大気–海洋 CO2 フラックス（バルク式）
  3. 生物ポンプ炭素輸送（温度依存 Michaelis-Menten 型）
  4. OTU エアレーション駆動力（垂直混合係数 κ の強制増加）

依存ライブラリ: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# 1. 定数・パラメータ
# ─────────────────────────────────────────
# 物理定数
RHO_SW   = 1025.0      # 海水密度 [kg/m³]
CP_SW    = 3994.0      # 海水比熱 [J/(kg·K)]
KH_CO2   = 3.4e-2      # CO2 ヘンリー定数 [mol/(L·atm)] at 25°C (近似)
PIS_CO2  = 420e-6      # 大気 CO2 分圧 [atm] (2026 年推定値)
K_GAS    = 0.08        # ガス交換係数 [mol/(m²·day·μatm)]

# 海洋層の深さ
H_SURF   = 100.0       # 表層混合層厚 [m]
H_DEEP   = 900.0       # 深層厚 [m]

# 通常時の垂直混合係数
KAPPA_BASE = 1.0e-5    # [m²/s]

# 生物ポンプパラメータ
V_MAX    = 0.5         # 最大炭素輸送速度 [Pg C/yr, normalized per unit area]
K_M      = 5.0         # 半飽和定数（栄養塩代理）[μmol/L]
T_OPT    = 22.0        # 生物ポンプ最適温度 [°C]
SIGMA_T  = 8.0         # 温度許容幅 [K]

# ベースライン条件（通常年）
T_SURF_0 = 24.0        # 表層温度 [°C]
T_DEEP_0 = 4.0         # 深層温度 [°C]
DIC_SURF_0 = 2000.0    # 表層溶存無機炭素 [μmol/kg]
DIC_DEEP_0 = 2200.0    # 深層 DIC [μmol/kg]
NUT_SURF_0 = 2.0       # 表層栄養塩 [μmol/L]
NUT_DEEP_0 = 30.0      # 深層栄養塩 [μmol/L]

# スーパーエルニーニョ熱強制（異常加熱）
T_ANOM_PEAK = 4.0      # ピーク表層温度偏差 [K]
T_ANOM_TAU  = 180.0    # 熱暴走の e-fold 時定数 [day]


# ─────────────────────────────────────────
# 2. OTU エアレーション強度スケジュール
# ─────────────────────────────────────────
def otu_kappa(t: float, otu_start: float, otu_intensity: float) -> float:
    """
    OTU 稼働後の垂直混合係数 κ を返す。
    稼働前はベースライン値、稼働後は段階的に増加。

    Parameters
    ----------
    t             : 現在時刻 [day]
    otu_start     : OTU 稼働開始時刻 [day]
    otu_intensity : κ の最大増加倍率 [-]
    """
    if t < otu_start:
        return KAPPA_BASE
    ramp = min(1.0, (t - otu_start) / 30.0)   # 30 日かけてフルパワーへ
    return KAPPA_BASE * (1.0 + (otu_intensity - 1.0) * ramp)


# ─────────────────────────────────────────
# 3. 強制項（エルニーニョ異常加熱）
# ─────────────────────────────────────────
def elnino_forcing(t: float, peak: float = T_ANOM_PEAK, tau: float = T_ANOM_TAU) -> float:
    """ガウス型熱暴走 [K] — ピークは t=tau day"""
    return peak * np.exp(-0.5 * ((t - tau) / (tau * 0.4)) ** 2)


# ─────────────────────────────────────────
# 4. 生物ポンプ炭素フラックス
# ─────────────────────────────────────────
def bio_pump_flux(T_s: float, nut: float) -> float:
    """
    温度・栄養塩依存の生物ポンプ炭素輸送 [相対単位]
    ガウス型温度依存 × Michaelis-Menten 型栄養塩依存
    """
    temp_factor = np.exp(-0.5 * ((T_s - T_OPT) / SIGMA_T) ** 2)
    nut_factor  = nut / (K_M + nut)
    return V_MAX * temp_factor * nut_factor


# ─────────────────────────────────────────
# 5. 海洋–大気 CO2 フラックス
# ─────────────────────────────────────────
def air_sea_co2_flux(DIC_s: float, T_s: float) -> float:
    """
    バルク式: F = k_gas * (pCO2_ocean - pCO2_atm)
    pCO2_ocean は DIC と温度から近似。
    正値 = 海洋から大気へ放出（吸収源弱体化）
    """
    # 温度補正ヘンリー定数（van't Hoff 近似）
    kh_T = KH_CO2 * np.exp(-2400.0 * (1.0 / (T_s + 273.15) - 1.0 / 298.15))
    pCO2_ocean = DIC_s / (kh_T * 1e6)   # 簡易近似 [atm]
    return K_GAS * (pCO2_ocean - PIS_CO2) * 1e6   # [μmol/(m²·day)]


# ─────────────────────────────────────────
# 6. ODE システム
# ─────────────────────────────────────────
def otu_system(t, y, otu_start, otu_intensity):
    """
    状態ベクトル y = [T_s, T_d, DIC_s, DIC_d, nut_s, nut_d]
    T_s  : 表層温度 [°C]
    T_d  : 深層温度 [°C]
    DIC_s: 表層 DIC [μmol/kg]
    DIC_d: 深層 DIC [μmol/kg]
    nut_s: 表層栄養塩 [μmol/L]
    nut_d: 深層栄養塩 [μmol/L]
    """
    T_s, T_d, DIC_s, DIC_d, nut_s, nut_d = y

    kappa = otu_kappa(t, otu_start, otu_intensity)
    dt_mix = kappa / (H_SURF * H_DEEP / (H_SURF + H_DEEP))  # 混合タイムスケール [1/s]
    dt_mix_day = dt_mix * 86400.0                             # [1/day]

    # ── 温度方程式 ──
    Q_elnino = elnino_forcing(t)             # 外部熱強制 [K/day, simplified]
    dTs = (Q_elnino
           - dt_mix_day * (T_s - T_d))
    dTd = dt_mix_day * (H_SURF / H_DEEP) * (T_s - T_d)

    # ── DIC 方程式 ──
    F_co2    = air_sea_co2_flux(DIC_s, T_s)   # 海洋→大気フラックス [μmol/(m²·day)]
    F_bio    = bio_pump_flux(T_s, nut_s)       # 生物ポンプ輸送 [相対]
    mix_DIC  = dt_mix_day * (DIC_d - DIC_s)

    dDICs = (- F_co2 / H_SURF              # 大気放出による表層 DIC 減少
             - F_bio                       # 生物ポンプによる深層輸送
             + mix_DIC)                    # 垂直混合
    dDICd = (+ F_bio * (H_SURF / H_DEEP)
             - dt_mix_day * (H_SURF / H_DEEP) * (DIC_d - DIC_s))

    # ── 栄養塩方程式 ──
    mix_nut  = dt_mix_day * (nut_d - nut_s)
    uptake   = F_bio * 0.1                    # 生物生産による栄養塩消費（レッドフィールド比近似）
    dNuts    = mix_nut - uptake
    dNutd    = - dt_mix_day * (H_SURF / H_DEEP) * (nut_d - nut_s) + uptake * (H_SURF / H_DEEP)

    return [dTs, dTd, dDICs, dDICd, dNuts, dNutd]


# ─────────────────────────────────────────
# 7. シミュレーション実行
# ─────────────────────────────────────────
def run_simulation(
    t_span_days: tuple = (0, 730),
    otu_start: float = 120.0,
    otu_intensity: float = 50.0,
    run_label: str = "OTU あり"
):
    """
    Parameters
    ----------
    t_span_days   : シミュレーション期間 [day]
    otu_start     : OTU 稼働開始日 [day from t=0]
    otu_intensity : κ の最大増加倍率（1 = OTU なし）
    run_label     : 図中ラベル
    """
    y0 = [T_SURF_0, T_DEEP_0, DIC_SURF_0, DIC_DEEP_0, NUT_SURF_0, NUT_DEEP_0]
    t_eval = np.linspace(t_span_days[0], t_span_days[1], 1460)

    sol = solve_ivp(
        fun=lambda t, y: otu_system(t, y, otu_start, otu_intensity),
        t_span=t_span_days,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    return sol, run_label


def compute_carbon_sink_efficiency(sol) -> np.ndarray:
    """生物ポンプ効率を正規化して炭素吸収源強度として返す（ベースライン=1.0）"""
    T_s   = sol.y[0]
    nut_s = sol.y[4]
    flux  = np.array([bio_pump_flux(T, n) for T, n in zip(T_s, nut_s)])
    baseline = bio_pump_flux(T_SURF_0, NUT_SURF_0)
    return flux / baseline


# ─────────────────────────────────────────
# 8. 可視化
# ─────────────────────────────────────────
def plot_results(scenarios: list):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "OTU 深海エアレーション強制駆動モデル\n"
        "チリ沖・赤道域スーパーエルニーニョ炭素吸収源喪失シミュレーション",
        fontsize=13
    )

    for sol, label in scenarios:
        t = sol.t
        T_s   = sol.y[0]
        DIC_s = sol.y[2]
        sink  = compute_carbon_sink_efficiency(sol)

        axes[0].plot(t, T_s, label=label)
        axes[1].plot(t, DIC_s, label=label)
        axes[2].plot(t, sink * 100.0, label=label)

    axes[0].axhline(T_SURF_0, color="gray", ls="--", lw=0.8, label="ベースライン")
    axes[0].set_ylabel("表層温度 [°C]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(DIC_SURF_0, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("表層 DIC [μmol/kg]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(100.0, color="gray", ls="--", lw=0.8, label="ベースライン (100%)")
    axes[2].axhline(90.0,  color="red",  ls=":",  lw=1.0, label="10% 喪失ライン")
    axes[2].set_ylabel("炭素吸収源効率 [%]")
    axes[2].set_xlabel("経過日数 [day]")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("otu_simulation_result.png", dpi=150, bbox_inches="tight")
    print("図を otu_simulation_result.png に保存しました。")
    plt.show()


# ─────────────────────────────────────────
# 9. エントリポイント
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("OTU 深海エアレーション強制駆動シミュレーション")
    print("スーパーエルニーニョ炭素吸収源喪失 10% 防止モデル")
    print("=" * 60)

    # シナリオ A: OTU なし（エルニーニョ放置）
    sol_no_otu, lbl_no = run_simulation(
        otu_start=9999,
        otu_intensity=1.0,
        run_label="OTU なし（無介入）"
    )

    # シナリオ B: OTU 中規模介入（κ × 20）
    sol_mid, lbl_mid = run_simulation(
        otu_start=120.0,
        otu_intensity=20.0,
        run_label="OTU 中規模 (κ×20, Day120 開始)"
    )

    # シナリオ C: OTU 大規模介入（κ × 50）
    sol_full, lbl_full = run_simulation(
        otu_start=120.0,
        otu_intensity=50.0,
        run_label="OTU 大規模 (κ×50, Day120 開始)"
    )

    # サマリー出力
    for sol, lbl in [(sol_no_otu, lbl_no), (sol_mid, lbl_mid), (sol_full, lbl_full)]:
        sink_min = compute_carbon_sink_efficiency(sol).min() * 100.0
        t_s_max  = sol.y[0].max()
        print(f"\n[{lbl}]")
        print(f"  表層温度 最大値    : {t_s_max:.2f} °C")
        print(f"  炭素吸収源効率 最低: {sink_min:.1f} %")
        recovered = compute_carbon_sink_efficiency(sol)[-1] * 100.0
        print(f"  最終日の吸収源効率 : {recovered:.1f} %")

    plot_results([
        (sol_no_otu, lbl_no),
        (sol_mid,    lbl_mid),
        (sol_full,   lbl_full),
    ])
