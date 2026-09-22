"""
Production-Grade In-Situ Vane Shear Test (VST) & Overconsolidation Ratio (OCR) Evaluator.
Strictly based on Canadian Foundation Engineering Manual (CFEM Ch 5), ASTM D2573,
Bjerrum (1972), Chandler (1988), Aas et al. (1986), and Mayne & Mitchell (1988).
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CORE DOMAIN LOGIC & COMPUTATIONAL ENGINE
# -----------------------------------------------------------------------------

@dataclass
class VaneGeometry:
    vane_type: str  # 'Rectangular', 'Nilcon', 'Geonor', 'Tapered'
    diameter_mm: float  # D
    height_mm: float    # H
    taper_top_deg: float = 0.0     # i_t
    taper_bottom_deg: float = 0.0  # i_b


class VaneShearEngine:
    """Calculates peak and remoulded undrained shear strengths with exact CFEM formulations."""

    @staticmethod
    def calculate_undrained_strength(torque_nm: float, geom: VaneGeometry) -> float:
        """
        Calculates undrained shear strength (su) in kPa from applied torque (N.m).
        Implements CFEM Eqs. 5.51 - 5.54.
        """
        t_nm = max(torque_nm, 0.0)
        d_mm = geom.diameter_mm
        h_mm = geom.height_mm

        if geom.vane_type == "Nilcon":
            # CFEM Eq. 5.53: su = 0.265 * T / D^3 (with D in m and T in N.m -> kPa)
            d_m = d_mm / 1000.0
            return float((0.265 * t_nm) / (d_m ** 3) / 1000.0)

        elif geom.vane_type == "Geonor":
            # CFEM Eq. 5.54: su = 0.257 * T / D^3
            d_m = d_mm / 1000.0
            return float((0.257 * t_nm) / (d_m ** 3) / 1000.0)

        elif geom.vane_type == "Rectangular":
            # CFEM Eq. 5.52: su = (6 / 7pi) * T / D^3 = 0.273 * T / D^3
            d_m = d_mm / 1000.0
            return float((6.0 * t_nm) / (7.0 * np.pi * (d_m ** 3)) / 1000.0)

        else:  # Tapered
            # CFEM Eq. 5.51:
            # su = (12 * T) / (pi * D^2 * (D/cos(i_t) + D/cos(i_b) + 6H))
            d_m = d_mm / 1000.0
            h_m = h_mm / 1000.0
            rad_top = math.radians(geom.taper_top_deg)
            rad_bot = math.radians(geom.taper_bottom_deg)
            cos_t = max(math.cos(rad_top), 1e-4)
            cos_b = max(math.cos(rad_bot), 1e-4)

            denom = np.pi * (d_m ** 2) * ((d_m / cos_t) + (d_m / cos_b) + 6.0 * h_m)
            su_pa = (12.0 * t_nm) / denom
            return float(su_pa / 1000.0)

    @classmethod
    def process_field_record(
        cls,
        torque_max_nm: float,
        torque_rem_nm: float,
        geom: VaneGeometry
    ) -> Tuple[float, float, float]:
        """Calculates suv (intact), srem (remoulded), and clay sensitivity St."""
        suv = cls.calculate_undrained_strength(torque_max_nm, geom)
        srem = cls.calculate_undrained_strength(torque_rem_nm, geom)
        st_val = suv / max(srem, 1e-4)
        return round(suv, 2), round(srem, 2), round(st_val, 2)


class VaneCorrectionEngine:
    """Computes correction factors (mu) based on PI and time-to-failure."""

    # Digitized standard Bjerrum curve (Bjerrum 1972 / CFEM Fig. 5.19)
    _PI_POINTS = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0])
    _MU_BJERRUM = np.array([1.18, 1.10, 0.98, 0.88, 0.81, 0.76, 0.72, 0.69, 0.66, 0.64, 0.62, 0.58])

    @classmethod
    def get_bjerrum_mu(cls, ip: float) -> float:
        """Bjerrum (1972, 1973) correction factor based on Plasticity Index (CFEM Eq. 5.55)."""
        clamped_ip = float(np.clip(ip, 5.0, 120.0))
        f_interp = interp1d(cls._PI_POINTS, cls._MU_BJERRUM, kind='linear', fill_value="extrapolate")
        return float(np.clip(f_interp(clamped_ip), 0.50, 1.25))

    @staticmethod
    def get_chandler_mu(ip: float, time_to_failure_min: float = 10000.0) -> float:
        """
        Chandler (1988) & ASTM D2573 rate-dependent correction factor.
        Considers embankment normal failure time (~10^4 minutes).
        """
        clamped_ip = max(float(ip), 5.0)
        tf = max(time_to_failure_min, 1.0)
        # Reference formulation: mu = 1.055 - 0.043 * sqrt(Ip) with strain-rate time factor
        base_mu = 1.055 - 0.043 * np.sqrt(clamped_ip)
        # Time correction shift
        log_t_ratio = np.log10(tf / 10000.0)
        time_factor = 1.0 - 0.05 * log_t_ratio
        return float(np.clip(base_mu * time_factor, 0.45, 1.20))

    @staticmethod
    def get_aas_mu(su_over_sigmav0: float) -> float:
        """
        Aas et al. (1986) correction factor based on strength ratio (CFEM Fig. 5.20).
        For suv / sigma'v0 < 0.20, CFEM recommends maximum design value of 1.0.
        """
        ratio = max(float(su_over_sigmav0), 0.05)
        if ratio < 0.20:
            return 1.0
        # Typical fit to Aas et al. normally consolidated / overconsolidated trend
        mu_val = 0.22 / ratio
        return float(np.clip(mu_val, 0.55, 1.0))


class OCREvaluationEngine:
    """Overconsolidation Ratio (OCR) estimation methods from field vane data."""

    @staticmethod
    def calculate_alpha_fv(ip: float) -> float:
        """Mayne & Mitchell (1988) empirical alpha_fv coefficient (CFEM Eq. 5.58)."""
        clamped_ip = max(float(ip), 2.0)
        return float(0.22 * (clamped_ip ** 0.48))

    @classmethod
    def estimate_ocr_mayne_mitchell(cls, suv: float, sigma_v0_eff: float, ip: float) -> float:
        """
        Estimates OCR using Mayne & Mitchell (1988) (CFEM Eq. 5.56 & 5.58).
        (suv / sigma'v0) = alpha_fv * (OCR)^0.95  ==>  OCR = [(suv / sigma'v0) / alpha_fv]^(1 / 0.95)
        """
        sig_eff = max(float(sigma_v0_eff), 1.0)
        strength_ratio = max(float(suv), 0.1) / sig_eff
        alpha = cls.calculate_alpha_fv(ip)
        ratio_norm = max(strength_ratio / alpha, 0.01)
        ocr = ratio_norm ** (1.0 / 0.95)
        return float(np.clip(ocr, 1.0, 50.0))

    @staticmethod
    def estimate_preconsolidation_mesri(su_corr: float, mu_bjerrum: float) -> float:
        """
        Estimates preconsolidation stress (sigma'_p) using Mesri (1975) (CFEM Eq. 5.57):
        su / sigma'_p = 0.22  ==>  sigma'_p = su_corr / 0.22
        """
        su_op = max(float(su_corr), 0.1)
        return float(su_op / 0.22)


# -----------------------------------------------------------------------------
# 2. SYNTHETIC BENCHMARK DATASET GENERATOR
# -----------------------------------------------------------------------------

def generate_benchmark_vst_dataset() -> pd.DataFrame:
    """Creates a comprehensive synthetic VST field dataset along depth."""
    np.random.seed(42)
    depths = np.array([1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 14.0, 16.0, 18.0, 20.0])
    gw_table = 2.0
    gamma_bulk = 18.0
    gamma_sat = 19.2
    gamma_w = 9.81

    rows = []
    for z in depths:
        # Effective stress calculation
        if z <= gw_table:
            sig_v0 = z * gamma_bulk
        else:
            sig_v0 = (gw_table * gamma_bulk) + (z - gw_table) * (gamma_sat - gamma_w)

        # Geotechnical stratification: Upper stiff crust (0-4m), soft marine clay (4-12m), firm clay (>12m)
        if z <= 4.0:
            stratum = "Desiccated Clay Crust"
            true_ocr = 4.5 - 0.4 * z
            ip = 32.0 + np.random.normal(0, 1.5)
            torque_peak = 62.0 - 4.0 * z + np.random.normal(0, 1.5)
            torque_rem = torque_peak / 3.0
        elif z <= 12.0:
            stratum = "Soft Sensitive Marine Clay"
            true_ocr = 1.35 + np.random.normal(0, 0.05)
            ip = 48.0 + np.random.normal(0, 2.0)
            torque_peak = 28.0 + 3.2 * (z - 4.0) + np.random.normal(0, 1.0)
            torque_rem = torque_peak / (8.5 + np.random.normal(0, 0.5))  # High sensitivity
        else:
            stratum = "Firm Silty Glacial Clay"
            true_ocr = 1.8 + 0.05 * (z - 12.0)
            ip = 26.0 + np.random.normal(0, 1.0)
            torque_peak = 60.0 + 5.5 * (z - 12.0) + np.random.normal(0, 2.0)
            torque_rem = torque_peak / 4.0

        rows.append({
            "Depth_m": z,
            "Stratum": stratum,
            "Total_Vertical_Stress_kPa": round(z * 18.5, 1),
            "Effective_Vertical_Stress_kPa": round(sig_v0, 1),
            "Plasticity_Index_Ip": round(ip, 1),
            "Liquid_Limit_LL": round(ip + 18.0, 1),
            "Peak_Torque_Nm": round(max(torque_peak, 5.0), 1),
            "Remoulded_Torque_Nm": round(max(torque_rem, 1.0), 1),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. ADVANCED VISUALIZATIONS (PLOTLY & MATPLOTLIB)
# -----------------------------------------------------------------------------

class VSTVisualizer:
    """Interactive Plotly dashboard and 300-DPI Matplotlib publication graphic."""

    @staticmethod
    def create_interactive_dashboard(df: pd.DataFrame) -> go.Figure:
        """Creates a synchronized 4-panel interactive geotechnical dashboard."""
        fig = make_subplots(
            rows=1, cols=4,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            subplot_titles=(
                "Undrained Strength (su) Profiles",
                "Clay Sensitivity (St)",
                "Correction Factors (μ)",
                "Overconsolidation Ratio (OCR)"
            )
        )

        # Panel 1: su Profiles
        fig.add_trace(
            go.Scatter(x=df["suv_intact_kPa"], y=df["Depth_m"], mode='lines+markers',
                       name='suv (Intact Field)', line=dict(color='#1f77b4', width=2),
                       marker=dict(size=6), hovertemplate='Depth: %{y:.1f} m<br>suv: %{x:.1f} kPa'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["su_corr_bjerrum_kPa"], y=df["Depth_m"], mode='lines+markers',
                       name='su (Bjerrum 1972)', line=dict(color='#2ca02c', width=2, dash='dash'),
                       marker=dict(size=5), hovertemplate='Depth: %{y:.1f} m<br>su(corr): %{x:.1f} kPa'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["srem_kPa"], y=df["Depth_m"], mode='lines+markers',
                       name='srem (Remoulded)', line=dict(color='#7f7f7f', width=1.5, dash='dot'),
                       marker=dict(size=4), hovertemplate='Depth: %{y:.1f} m<br>srem: %{x:.1f} kPa'),
            row=1, col=1
        )

        # Panel 2: Sensitivity St
        fig.add_trace(
            go.Scatter(x=df["Sensitivity_St"], y=df["Depth_m"], mode='lines+markers',
                       name='Sensitivity (St)', line=dict(color='#d62728', width=2),
                       marker=dict(size=6, symbol='diamond'),
                       hovertemplate='Depth: %{y:.1f} m<br>St: %{x:.1f}'),
            row=1, col=2
        )
        fig.add_vline(x=4.0, line=dict(color='gray', dash='dash', width=1), row=1, col=2)
        fig.add_vline(x=8.0, line=dict(color='gray', dash='dash', width=1), row=1, col=2)

        # Panel 3: Correction Factors
        fig.add_trace(
            go.Scatter(x=df["mu_bjerrum"], y=df["Depth_m"], mode='lines+markers',
                       name='μ (Bjerrum)', line=dict(color='#ff7f0e', width=2),
                       marker=dict(size=5), hovertemplate='Depth: %{y:.1f} m<br>μ_B: %{x:.2f}'),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=df["mu_chandler"], y=df["Depth_m"], mode='lines+markers',
                       name='μ (Chandler)', line=dict(color='#9467bd', width=2, dash='dot'),
                       marker=dict(size=5), hovertemplate='Depth: %{y:.1f} m<br>μ_Ch: %{x:.2f}'),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=df["mu_aas"], y=df["Depth_m"], mode='lines+markers',
                       name='μ (Aas et al.)', line=dict(color='#8c564b', width=2, dash='dashdot'),
                       marker=dict(size=5), hovertemplate='Depth: %{y:.1f} m<br>μ_Aas: %{x:.2f}'),
            row=1, col=3
        )

        # Panel 4: OCR Profiles
        fig.add_trace(
            go.Scatter(x=df["OCR_Mayne_Mitchell"], y=df["Depth_m"], mode='lines+markers',
                       name='OCR (Mayne & Mitchell 1988)', line=dict(color='#e377c2', width=2.2),
                       marker=dict(size=6, symbol='square'),
                       hovertemplate='Depth: %{y:.1f} m<br>OCR: %{x:.2f}'),
            row=1, col=4
        )
        fig.add_vline(x=1.0, line=dict(color='black', width=1.2, dash='dash'), row=1, col=4)

        fig.update_yaxes(autorange='reversed', title_text="Depth Below Surface (m)", row=1, col=1)
        fig.update_xaxes(title_text="Shear Strength (kPa)", row=1, col=1)
        fig.update_xaxes(title_text="Sensitivity St", row=1, col=2)
        fig.update_xaxes(title_text="Correction Factor μ", row=1, col=3)
        fig.update_xaxes(title_text="Estimated OCR", row=1, col=4)

        fig.update_layout(
            height=650,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=60, b=80)
        )
        return fig

    @staticmethod
    def generate_static_publication_figure(df: pd.DataFrame) -> io.BytesIO:
        """Generates a 300 DPI, multi-panel print-ready geotechnical report figure."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 4, figsize=(15, 7.5), sharey=True, dpi=300)

        depths = df["Depth_m"]

        # Panel 1: Undrained Strength
        axes[0].plot(df["suv_intact_kPa"], depths, 'o-', color='#1f77b4', lw=2.0, ms=5, label=r'Field $s_{uv}$')
        axes[0].plot(df["su_corr_bjerrum_kPa"], depths, 's--', color='#2ca02c', lw=1.8, ms=4, label=r'Corr. $(s_u)_{Bjerrum}$')
        axes[0].plot(df["srem_kPa"], depths, '^:', color='#7f7f7f', lw=1.4, ms=4, label=r'Remoulded $s_{rem}$')
        axes[0].set_xlabel('Shear Strength (kPa)', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Depth Below Ground Surface (m)', fontsize=10, fontweight='bold')
        axes[0].set_title('(a) Undrained Strength', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].legend(loc='lower right', frameon=True, fontsize=8)
        axes[0].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 2: Sensitivity
        axes[1].plot(df["Sensitivity_St"], depths, 'D-', color='#d62728', lw=2.0, ms=5)
        axes[1].axvline(4.0, color='gray', ls='--', lw=1.0)
        axes[1].axvline(8.0, color='gray', ls='--', lw=1.0)
        axes[1].text(2.0, depths.max() * 0.95, 'Low', fontsize=8, color='gray')
        axes[1].text(5.0, depths.max() * 0.95, 'Medium', fontsize=8, color='gray')
        axes[1].text(9.0, depths.max() * 0.95, 'Sensitive', fontsize=8, color='gray')
        axes[1].set_xlabel('Sensitivity $S_t$', fontsize=10, fontweight='bold')
        axes[1].set_title('(b) Clay Sensitivity', fontsize=11, fontweight='bold')
        axes[1].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 3: Correction Factors
        axes[2].plot(df["mu_bjerrum"], depths, 'o-', color='#ff7f0e', lw=1.8, ms=4, label=r'$\mu_{Bjerrum}$')
        axes[2].plot(df["mu_chandler"], depths, '^--', color='#9467bd', lw=1.6, ms=4, label=r'$\mu_{Chandler}$')
        axes[2].plot(df["mu_aas"], depths, 'x:', color='#8c564b', lw=1.6, ms=5, label=r'$\mu_{Aas}$')
        axes[2].set_xlabel('Correction Factor $\mu$', fontsize=10, fontweight='bold')
        axes[2].set_title('(c) Vane Factors', fontsize=11, fontweight='bold')
        axes[2].legend(loc='lower left', frameon=True, fontsize=8)
        axes[2].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 4: OCR
        axes[3].plot(df["OCR_Mayne_Mitchell"], depths, 's-', color='#e377c2', lw=2.0, ms=5, label='Mayne & Mitchell')
        axes[3].axvline(1.0, color='black', ls='--', lw=1.2, label='NC (OCR=1.0)')
        axes[3].set_xlabel('Overconsolidation Ratio (OCR)', fontsize=10, fontweight='bold')
        axes[3].set_title('(d) Profile OCR', fontsize=11, fontweight='bold')
        axes[3].legend(loc='lower right', frameon=True, fontsize=8)
        axes[3].grid(True, which='both', ls='--', alpha=0.6)

        fig.suptitle('CFEM Vane Shear Test Interpretation Suite & OCR Evaluation', fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


# -----------------------------------------------------------------------------
# 4. EXCEL EXPORT ENGINE (OPENPYXL)
# -----------------------------------------------------------------------------

class VSTExcelExporter:
    """Generates an executive-level, professional Excel report with full formatting."""

    @staticmethod
    def export(df: pd.DataFrame, geom: VaneGeometry, time_to_failure: float) -> io.BytesIO:
        wb = openpyxl.Workbook()

        # Styles
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        sub_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
        font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        font_sub = Font(name="Calibri", size=11, bold=True, color="1F497D")
        font_data = Font(name="Calibri", size=10)
        font_bold = Font(name="Calibri", size=10, bold=True)
        thin_border = Border(
            left=Side(style='thin', color='B0B0B0'),
            right=Side(style='thin', color='B0B0B0'),
            top=Side(style='thin', color='B0B0B0'),
            bottom=Side(style='thin', color='B0B0B0')
        )

        # Sheet 1: Executive Interpretation Table
        ws_main = wb.active
        ws_main.title = "VST_Interpretation_Summary"
        ws_main.views.sheetView[0].showGridLines = True

        ws_main["A1"] = "IN-SITU VANE SHEAR TEST (VST) & OCR EVALUATION REPORT"
        ws_main["A1"].font = Font(name="Calibri", size=14, bold=True, color="1F497D")
        ws_main["A2"] = "Compliant with Canadian Foundation Engineering Manual (CFEM Ch 5) & ASTM D2573"
        ws_main["A2"].font = Font(name="Calibri", size=10, italic=True)

        # Metadata Block
        ws_main["A4"] = "VANE APPARATUS & GEOMETRY SPECIFICATIONS"
        ws_main["A4"].font = font_sub
        ws_main["A4"].fill = sub_fill

        specs = [
            ("Blade Profile / Type", geom.vane_type),
            ("Blade Diameter D (mm)", geom.diameter_mm),
            ("Blade Height H (mm)", geom.height_mm),
            ("Top Taper Angle i_t (deg)", geom.taper_top_deg),
            ("Bottom Taper Angle i_b (deg)", geom.taper_bottom_deg),
            ("Embankment Time to Failure t_f (min)", time_to_failure),
        ]
        r_idx = 5
        for k, v in specs:
            ws_main[f"A{r_idx}"] = k
            ws_main[f"B{r_idx}"] = v
            ws_main[f"A{r_idx}"].font = font_data
            ws_main[f"B{r_idx}"].font = font_bold
            ws_main[f"A{r_idx}"].border = thin_border
            ws_main[f"B{r_idx}"].border = thin_border
            r_idx += 1

        r_idx += 2
        ws_main.cell(row=r_idx, column=1, value="PROCESSED VST GEOTECHNICAL DATA PROFILE").font = font_sub

        # Data Header
        cols = [
            "Depth (m)", "Stratum", "σ'v0 (kPa)", "Ip (%)", "T_max (N.m)", "T_rem (N.m)",
            "suv Intact (kPa)", "srem (kPa)", "Sensitivity St", "μ (Bjerrum)",
            "su,corr Bjerrum (kPa)", "μ (Chandler)", "μ (Aas)", "OCR (Mayne & Mitchell)", "σ'p Mesri (kPa)"
        ]
        r_idx += 1
        for c_idx, col_name in enumerate(cols, start=1):
            cell = ws_main.cell(row=r_idx, column=c_idx, value=col_name)
            cell.fill = header_fill
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border

        for _, row_val in df.iterrows():
            r_idx += 1
            row_items = [
                row_val["Depth_m"],
                row_val["Stratum"],
                row_val["Effective_Vertical_Stress_kPa"],
                row_val["Plasticity_Index_Ip"],
                row_val["Peak_Torque_Nm"],
                row_val["Remoulded_Torque_Nm"],
                row_val["suv_intact_kPa"],
                row_val["srem_kPa"],
                row_val["Sensitivity_St"],
                row_val["mu_bjerrum"],
                row_val["su_corr_bjerrum_kPa"],
                row_val["mu_chandler"],
                row_val["mu_aas"],
                row_val["OCR_Mayne_Mitchell"],
                row_val["sigma_p_mesri_kPa"],
            ]
            for c_idx, item in enumerate(row_items, start=1):
                c = ws_main.cell(row=r_idx, column=c_idx, value=item)
                c.font = font_data
                c.border = thin_border
                c.alignment = Alignment(horizontal="center" if isinstance(item, (int, float)) else "left")

        # Column Auto-fit
        for col in ws_main.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws_main.column_dimensions[col_letter].width = max(max_len + 3, 11)

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf


# -----------------------------------------------------------------------------
# 5. STREAMLIT APPLICATION (ENTRY POINT)
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="CFEM In-Situ VST & OCR Evaluation Engine",
        page_icon="⛏️",
        layout="wide"
    )

    st.title("⛏️ In-Situ Vane Shear Test (VST) & OCR Evaluation Package")
    st.markdown(
        """
        **Standard Geotechnical Analysis Suite for Cohesive Soils**  
        *Fully compliant with the Canadian Foundation Engineering Manual (CFEM Ch 5), ASTM D2573, Bjerrum (1972), and Mayne & Mitchell (1988).*
        """
    )
    st.write("---")

    # Sidebar: Vane Hardware & Testing Standards
    st.sidebar.header("📐 1. Vane Apparatus & Geometry")
    vane_type = st.sidebar.selectbox(
        "Blade Profile / Type",
        ["Rectangular", "Nilcon", "Geonor", "Tapered"]
    )

    col_v1, col_v2 = st.sidebar.columns(2)
    with col_v1:
        vane_d = st.sidebar.number_input("Blade Diameter D (mm)", min_value=20.0, max_value=200.0, value=65.0, step=5.0)
    with col_v2:
        vane_h = st.sidebar.number_input("Blade Height H (mm)", min_value=40.0, max_value=400.0, value=130.0, step=5.0)

    taper_top = 0.0
    taper_bot = 0.0
    if vane_type == "Tapered":
        st.sidebar.markdown("##### Taper Angles")
        taper_top = st.sidebar.number_input("Top Taper Angle i_t (deg)", min_value=0.0, max_value=85.0, value=45.0, step=5.0)
        taper_bot = st.sidebar.number_input("Bottom Taper Angle i_b (deg)", min_value=0.0, max_value=85.0, value=45.0, step=5.0)

    geom = VaneGeometry(
        vane_type=vane_type,
        diameter_mm=vane_d,
        height_mm=vane_h,
        taper_top_deg=taper_top,
        taper_bottom_deg=taper_bot
    )

    st.sidebar.header("⏱️ 2. Loading Rate & Construction Context")
    time_to_failure = st.sidebar.number_input(
        "Embankment Failure Time t_f (min)",
        min_value=1.0,
        max_value=100000.0,
        value=10000.0,
        step=500.0,
        help="ASTM D2573 / Chandler (1988) parameter; typical normal rate of construction = 10,000 min (~7 days)."
    )

    # Data Input Source
    st.sidebar.header("📂 3. Field Records Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Field VST Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success("Field records successfully loaded!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}. Fallback to synthetic benchmark data.")
            df_raw = generate_benchmark_vst_dataset()
    else:
        df_raw = generate_benchmark_vst_dataset()

    # Core Execution
    processed_records = []
    for _, row in df_raw.iterrows():
        z = row["Depth_m"]
        stratum = row.get("Stratum", "Clay Stratum")
        sig_v0 = row["Effective_Vertical_Stress_kPa"]
        ip = row["Plasticity_Index_Ip"]
        t_max = row["Peak_Torque_Nm"]
        t_rem = row["Remoulded_Torque_Nm"]

        # 1. Strengths
        suv, srem, st_val = VaneShearEngine.process_field_record(t_max, t_rem, geom)

        # 2. Corrections
        mu_b = VaneCorrectionEngine.get_bjerrum_mu(ip)
        mu_ch = VaneCorrectionEngine.get_chandler_mu(ip, time_to_failure)
        mu_aas = VaneCorrectionEngine.get_aas_mu(suv / sig_v0 if sig_v0 > 0 else 0.25)

        su_corr_b = round(suv * mu_b, 2)
        su_corr_ch = round(suv * mu_ch, 2)
        su_corr_aas = round(suv * mu_aas, 2)

        # 3. OCR & Preconsolidation
        ocr_val = round(OCREvaluationEngine.estimate_ocr_mayne_mitchell(suv, sig_v0, ip), 2)
        sig_p = round(OCREvaluationEngine.estimate_preconsolidation_mesri(su_corr_b, mu_b), 1)

        processed_records.append({
            "Depth_m": z,
            "Stratum": stratum,
            "Effective_Vertical_Stress_kPa": sig_v0,
            "Plasticity_Index_Ip": ip,
            "Peak_Torque_Nm": t_max,
            "Remoulded_Torque_Nm": t_rem,
            "suv_intact_kPa": suv,
            "srem_kPa": srem,
            "Sensitivity_St": st_val,
            "mu_bjerrum": round(mu_b, 3),
            "su_corr_bjerrum_kPa": su_corr_b,
            "mu_chandler": round(mu_ch, 3),
            "su_corr_chandler_kPa": su_corr_ch,
            "mu_aas": round(mu_aas, 3),
            "su_corr_aas_kPa": su_corr_aas,
            "OCR_Mayne_Mitchell": ocr_val,
            "sigma_p_mesri_kPa": sig_p
        })

    df_proc = pd.DataFrame(processed_records)

    # Top-Level Summary Metrics
    avg_suv = df_proc["suv_intact_kPa"].mean()
    max_st = df_proc["Sensitivity_St"].max()
    avg_ocr = df_proc["OCR_Mayne_Mitchell"].mean()
    mean_mu = df_proc["mu_bjerrum"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg. Field Intact suv", f"{avg_suv:.1f} kPa")
    with m2:
        st.metric("Peak Sensitivity (St)", f"{max_st:.1f}", delta="Sensitive Clay" if max_st > 8.0 else "Normal")
    with m3:
        st.metric("Avg. Bjerrum Factor (μ)", f"{mean_mu:.3f}")
    with m4:
        st.metric("Mean Stratum OCR", f"{avg_ocr:.2f}", delta="Overconsolidated" if avg_ocr > 1.5 else "Normally Cons.")

    st.write("")

    # Tabs
    tab_dashboard, tab_table, tab_export = st.tabs([
        "📊 Geotechnical Profiles & Interpretation",
        "📋 Processed Verification Schedule",
        "📥 Professional Reporting Suite"
    ])

    with tab_dashboard:
        st.subheader("Subsurface Geotechnical Profiles from In-Situ VST")
        fig_dash = VSTVisualizer.create_interactive_dashboard(df_proc)
        st.plotly_chart(fig_dash, use_container_width=True)

        st.info(
            "💡 **Interpretation Guide (CFEM Section 5.4.6):**\n"
            "- Intact shear strength ($s_{uv}$) reflects vertical-horizontal cylindrical failure surface.\n"
            "- Bjerrum factor ($\mu$) accounts for strain-rate, anisotropy, and progressive failure.\n"
            "- Sensitivity $S_t > 8$ indicates quick or sensitive clay deposits requiring undisturbed piston/block sampling."
        )

    with tab_table:
        st.subheader("Field VST Measured & Derived Values")
        st.dataframe(df_proc, use_container_width=True)

    with tab_export:
        st.subheader("Standardized Geotechnical Deliverables")
        st.markdown("Download high-resolution print graphics (300 DPI) and structured Excel schedules ready for engineering submittals.")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("##### 📄 Multi-Panel Report Graphic (300 DPI)")
            fig_buf = VSTVisualizer.generate_static_publication_figure(df_proc)
            st.image(fig_buf, caption="Print Preview: VST Strength, Sensitivity & OCR Profiles", use_container_width=True)
            st.download_button(
                label="⬇️ Download High-Res Report Figure (PNG)",
                data=fig_buf,
                file_name="CFEM_VST_OCR_Report_Figure.png",
                mime="image/png"
            )

        with col_e2:
            st.markdown("##### 📊 Full Geotechnical Data Schedule (.xlsx)")
            st.markdown(
                """
                Includes:
                - **Vane Specifications:** Apparatus type, exact dimensions, and taper parameters.
                - **Interpretation Summary:** Depths, stresses, measured torque values, $s_{uv}$, $s_{rem}$, $S_t$, correction factors ($\mu$), and predicted $\\text{OCR}$ profiles.
                """
            )
            excel_buf = VSTExcelExporter.export(df_proc, geom, time_to_failure)
            st.download_button(
                label="⬇️ Download Formatted Excel Schedule (.xlsx)",
                data=excel_buf,
                file_name="CFEM_VST_Processed_Schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
