"""
Generate all 3 figures for the El Niño Japanese scientific article.
No cartopy required — uses schematic matplotlib-only approach.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ─────────────────────────────────────────────────
# Shared font config (Japanese-compatible fallback)
# ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": ["Meiryo", "Yu Gothic", "MS Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# ══════════════════════════════════════════════════════════
# FIGURE 1 — World SST Anomaly Map (schematic)
# ══════════════════════════════════════════════════════════

def make_figure1():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # ── Ocean base ──
    ocean = plt.Rectangle((-180, -90), 360, 180, color="#0d2240", zorder=0)
    ax.add_patch(ocean)

    # ── Simple continental outlines (schematic polygons) ──
    continent_color = "#2d4a2d"
    # North America
    na = plt.Polygon([(-168,72),(-52,72),(-52,25),(-80,8),(-85,8),(-90,16),(-110,20),(-120,30),(-168,60)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(na)
    # South America
    sa = plt.Polygon([(-80,10),(-34,10),(-34,-56),(-70,-56),(-80,10)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(sa)
    # Europe
    eu = plt.Polygon([(-10,36),(40,36),(40,72),(-10,72)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(eu)
    # Africa
    af = plt.Polygon([(-18,-35),(52,-35),(52,38),(-18,38)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(af)
    # Asia
    as_ = plt.Polygon([(25,10),(145,10),(145,72),(25,72)],
                      closed=True, color=continent_color, zorder=2)
    ax.add_patch(as_)
    # Australia
    au = plt.Polygon([(114,-40),(154,-40),(154,-10),(114,-10)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(au)
    # Greenland
    gl = plt.Polygon([(-55,59),(-15,59),(-15,84),(-55,84)],
                     closed=True, color=continent_color, zorder=2)
    ax.add_patch(gl)

    # ── SST anomaly gradient — warm in central/eastern equatorial Pacific ──
    cmap_anom = LinearSegmentedColormap.from_list(
        "sst_anomaly",
        ["#1a3a8a", "#2255cc", "#44aaff", "#ffffff", "#ffcc00", "#ff6600", "#cc0000"],
        N=256,
    )

    # Build a 2D SST anomaly field
    lon = np.linspace(-180, 180, 1800)
    lat = np.linspace(-90, 90, 900)
    LON, LAT = np.meshgrid(lon, lat)

    # Warm anomaly centered on central/eastern equatorial Pacific
    anom = (
        3.5 * np.exp(-((LON - (-140)) ** 2) / 2500 - (LAT ** 2) / 180)
        + 2.5 * np.exp(-((LON - (-100)) ** 2) / 1800 - (LAT ** 2) / 120)
        + 1.5 * np.exp(-((LON - (-160)) ** 2) / 2000 - ((LAT - 8) ** 2) / 200)
        # Slight cool in western Pacific / Indian Ocean
        - 1.2 * np.exp(-((LON - 150) ** 2) / 800 - (LAT ** 2) / 200)
        - 0.8 * np.exp(-((LON - 80) ** 2) / 1200 - (LAT ** 2) / 300)
        # Cool near Peru coast (upwelling suppressed but underlying cool)
        - 0.4 * np.exp(-((LON - (-78)) ** 2) / 200 - ((LAT - (-10)) ** 2) / 80)
        # Global background warm anomaly
        + 0.3
    )
    anom = np.clip(anom, -2.5, 4.5)

    img = ax.imshow(
        anom,
        extent=[-180, 180, -90, 90],
        origin="lower",
        cmap=cmap_anom,
        vmin=-2.5,
        vmax=4.5,
        alpha=0.82,
        zorder=1,
        aspect="auto",
    )

    # ── Colorbar ──
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal",
                        pad=0.04, fraction=0.032, shrink=0.6)
    cbar.set_label("海面水温偏差 (°C)", color="white", fontsize=11)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_xticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("white")

    # ── Grid lines ──
    for lon_line in range(-180, 181, 30):
        ax.axvline(lon_line, color="white", lw=0.3, alpha=0.25, zorder=3)
    for lat_line in range(-90, 91, 30):
        ax.axhline(lat_line, color="white", lw=0.3, alpha=0.25, zorder=3)
    ax.axhline(0, color="white", lw=0.6, alpha=0.5, zorder=3, linestyle="--")

    # ── Annotation arrows & labels ──
    ax_kw = dict(color="white", fontsize=10.5, fontweight="bold", zorder=10,
                 ha="center",
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Main warm zone label
    ax.text(-135, 5, "中部〜東部赤道太平洋で高温", **ax_kw)
    ax.annotate("", xy=(-120, 2), xytext=(-145, 4),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8), zorder=10)

    # Trade wind label
    ax.text(-165, 18, "貿易風の弱化", color="#ffdd44", fontsize=10, fontweight="bold",
            zorder=10, ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    ax.annotate("", xy=(-148, 10), xytext=(-160, 16),
                arrowprops=dict(arrowstyle="->", color="#ffdd44", lw=1.6), zorder=10)

    # Warm water eastward
    ax.text(-100, -12, "暖水の東進 →", color="#ff8844", fontsize=10, fontweight="bold",
            zorder=10, ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Cool label (Indonesia)
    ax.text(130, -5, "西太平洋で\n低温・乾燥傾向", color="#88ddff", fontsize=9,
            zorder=10, ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Peru upwelling suppressed
    ax.text(-72, -20, "アップウェリング\n弱化", color="#aaddff", fontsize=9,
            zorder=10, ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Equator label
    ax.text(170, 2, "赤道", color="white", fontsize=8, zorder=10, alpha=0.7)

    # Axis labels
    ax.set_xticks(range(-180, 181, 30))
    ax.set_xticklabels([f"{abs(x)}°{'W' if x < 0 else ('E' if x > 0 else '')}"
                        for x in range(-180, 181, 30)],
                       color="white", fontsize=8)
    ax.set_yticks(range(-90, 91, 30))
    ax.set_yticklabels([f"{abs(y)}°{'S' if y < 0 else ('N' if y > 0 else '')}"
                        for y in range(-90, 91, 30)],
                       color="white", fontsize=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax.set_title(
        "図1　エルニーニョ発生時の世界の海面水温偏差（概念図）\n"
        "Fig. 1  Schematic World SST Anomaly During El Niño",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig("figures/el_nino_world_sst_anomaly_map.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Figure 1 saved.")


# ══════════════════════════════════════════════════════════
# FIGURE 2 — Normal vs El Niño Pacific Diagram (two-panel)
# ══════════════════════════════════════════════════════════

def make_figure2():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#f0f4f8")
    fig.subplots_adjust(wspace=0.04)

    ocean_deep = "#1a3f6e"
    ocean_warm = "#cc3300"
    ocean_mid = "#3377bb"
    land_col = "#7ab648"
    sky_col = "#ddeeff"

    def draw_panel(ax, is_el_nino):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_facecolor(sky_col)
        ax.set_aspect("equal")

        title = "エルニーニョ時" if is_el_nino else "平常時（ラニーニャ寄り）"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8, color="#1a1a2e")

        # ── Sky ──
        sky = plt.Rectangle((0, 3.8), 10, 2.2, color=sky_col, zorder=0)
        ax.add_patch(sky)

        # ── Ocean layers ──
        # Deep cool layer
        deep = plt.Rectangle((0, 0), 10, 2.2, color=ocean_deep, zorder=1)
        ax.add_patch(deep)

        if not is_el_nino:
            # Normal: warm water piled in the west, thin in east; upwelling near east
            warm_west = plt.Polygon(
                [(0, 2.2), (5.5, 2.2), (3, 3.6), (0, 3.6)],
                closed=True, color="#e84545", alpha=0.88, zorder=2
            )
            warm_east_thin = plt.Polygon(
                [(5.5, 2.2), (10, 2.2), (10, 2.6), (7, 2.6)],
                closed=True, color="#ff9955", alpha=0.6, zorder=2
            )
            ax.add_patch(warm_west)
            ax.add_patch(warm_east_thin)

            # Thermocline line
            thermo_x = [0, 3, 5.5, 7, 10]
            thermo_y = [3.6, 3.6, 2.2, 2.55, 2.6]
            ax.plot(thermo_x, thermo_y, color="cyan", lw=2.5, linestyle="--",
                    zorder=5, label="thermocline")
            ax.text(2.5, 3.75, "温度躍層（サーモクライン）",
                    fontsize=8, color="cyan", ha="center", zorder=6)

            # Upwelling arrows near east coast
            for y_start in [0.3, 0.8, 1.3, 1.8]:
                ax.annotate("", xy=(9.1, y_start + 0.6), xytext=(9.1, y_start),
                            arrowprops=dict(arrowstyle="->", color="#44ddff", lw=1.8), zorder=7)
            ax.text(9.0, 0.1, "アップウェリング\n（強い）",
                    fontsize=8.5, color="#44ddff", ha="center", zorder=8, fontweight="bold")

            # Trade wind arrows (west-blowing, strong)
            for x_pos in [7.5, 5.5, 3.5, 1.5]:
                ax.annotate("", xy=(x_pos - 1.2, 4.7), xytext=(x_pos, 4.7),
                            arrowprops=dict(arrowstyle="->", color="#336699", lw=2.5), zorder=7)
            ax.text(4.5, 5.2, "貿易風（強い）←", fontsize=10, color="#224488",
                    ha="center", fontweight="bold", zorder=8)

            # Walker circulation arrow
            ax.annotate("", xy=(1.5, 5.5), xytext=(8.5, 5.5),
                        arrowprops=dict(arrowstyle="fancy", color="#445577",
                                        connectionstyle="arc3,rad=0.4", lw=1.5), zorder=7)
            ax.text(5, 5.8, "ウォーカー循環（強い）", fontsize=8.5, color="#334466",
                    ha="center", zorder=8)

            # Labels: warm pool & cold tongue
            ax.text(1.8, 2.9, "暖水プール", fontsize=9.5, color="white",
                    fontweight="bold", ha="center", zorder=9)
            ax.text(8.5, 2.35, "冷水\n（湧昇）", fontsize=9, color="#cceeff",
                    ha="center", zorder=9)

        else:
            # El Niño: warm water spread east, thermocline flattened
            warm_flat = plt.Polygon(
                [(0, 2.2), (10, 2.2), (10, 3.3), (0, 3.5)],
                closed=True, color="#e84545", alpha=0.88, zorder=2
            )
            ax.add_patch(warm_flat)

            # Thermocline — much flatter
            thermo_x = [0, 5, 10]
            thermo_y = [3.5, 3.15, 3.3]
            ax.plot(thermo_x, thermo_y, color="cyan", lw=2.5, linestyle="--",
                    zorder=5)
            ax.text(5, 3.65, "温度躍層（平坦化）",
                    fontsize=8, color="cyan", ha="center", zorder=6)

            # Upwelling suppressed
            for y_start in [0.3, 0.8]:
                ax.annotate("", xy=(9.1, y_start + 0.35), xytext=(9.1, y_start),
                            arrowprops=dict(arrowstyle="->", color="#8899aa", lw=1.2,
                                            linestyle="dashed"), zorder=7)
            ax.text(9.1, 0.1, "アップウェリング\n（弱い）",
                    fontsize=8.5, color="#8899aa", ha="center", zorder=8)
            # X mark
            ax.text(9.4, 1.6, "✕", fontsize=18, color="#cc2222", ha="center", zorder=9)

            # Weak trade wind arrows
            for x_pos in [6.5, 4.5]:
                ax.annotate("", xy=(x_pos - 0.8, 4.7), xytext=(x_pos, 4.7),
                            arrowprops=dict(arrowstyle="->", color="#aabbcc", lw=1.5,
                                            linestyle="dashed"), zorder=7)
            ax.text(4.5, 5.2, "貿易風（弱い）", fontsize=10, color="#889aaa",
                    ha="center", fontweight="bold", zorder=8)

            # Walker circulation weakened
            ax.annotate("", xy=(3.0, 5.5), xytext=(7.5, 5.5),
                        arrowprops=dict(arrowstyle="fancy", color="#8899aa",
                                        connectionstyle="arc3,rad=0.4", lw=1.2,
                                        linestyle="dashed"), zorder=7)
            ax.text(5, 5.8, "ウォーカー循環（弱い）", fontsize=8.5, color="#8899aa",
                    ha="center", zorder=8)

            # Labels
            ax.text(4.5, 2.75, "暖水の東進・拡大", fontsize=9.5, color="white",
                    fontweight="bold", ha="center", zorder=9)
            ax.text(8.5, 2.75, "暖水\n（栄養塩不足）", fontsize=9, color="white",
                    ha="center", zorder=9)

        # ── Land blocks ──
        # West (Asia/Indonesia)
        land_w = plt.Rectangle((0, 3.6), 1.1, 1.0, color=land_col, zorder=3)
        ax.add_patch(land_w)
        ax.text(0.55, 4.9, "インドネシア\n/西太平洋", fontsize=7.5, ha="center",
                color="#1a3300", zorder=8)

        # East (South America)
        land_e = plt.Rectangle((8.9, 3.3 if is_el_nino else 2.6), 1.1, 1.0,
                                color=land_col, zorder=3)
        ax.add_patch(land_e)
        ax.text(9.45, 4.9, "南米\n/ペルー", fontsize=7.5, ha="center",
                color="#1a3300", zorder=8)

        # ── Deep ocean label ──
        ax.text(5, 1.1, "深層冷水（栄養塩豊富）",
                fontsize=9, color="#aaddff", ha="center", zorder=4)

        # Axis off
        ax.axis("off")

        # Border
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_visible(False)

    draw_panel(axes[0], is_el_nino=False)
    draw_panel(axes[1], is_el_nino=True)

    fig.suptitle(
        "図2　平常時とエルニーニョ時の太平洋の比較模式図\n"
        "Fig. 2  Schematic Comparison: Normal Conditions vs El Niño in the Pacific",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=1.01,
    )

    # Legend
    legend_elems = [
        mpatches.Patch(color="#e84545", label="暖水（高温）"),
        mpatches.Patch(color=ocean_deep, label="深層冷水"),
        plt.Line2D([0], [0], color="cyan", linestyle="--", lw=2, label="温度躍層"),
        plt.Line2D([0], [0], color="#44ddff", lw=2,
                   marker=">", label="アップウェリング（強）"),
        plt.Line2D([0], [0], color="#8899aa", lw=1.5, linestyle="--",
                   marker=">", label="アップウェリング（弱）"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.savefig("figures/normal_vs_el_nino_pacific_diagram.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Figure 2 saved.")


# ══════════════════════════════════════════════════════════
# FIGURE 3 — Ecosystem Impact Flowchart
# ══════════════════════════════════════════════════════════

def make_figure3():
    fig, ax = plt.subplots(figsize=(13, 16), facecolor="#f7f9fc")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    ax.set_facecolor("#f7f9fc")

    def draw_box(ax, x, y, w, h, text, color, text_color="white",
                 fontsize=10, radius=0.35, zorder=5, bold=True):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle=f"round,pad=0.1,rounding_size={radius}",
                             facecolor=color, edgecolor="white", linewidth=1.5,
                             zorder=zorder)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold" if bold else "normal",
                zorder=zorder + 1, wrap=True,
                multialignment="center")

    def arrow(ax, x1, y1, x2, y2, color="#555577", lw=2.0, zorder=4):
        ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=18),
                    zorder=zorder)

    # ── Color palette ──
    C_DRIVER  = "#1a3a6e"   # darkest — primary driver
    C_OCEAN1  = "#1a6ea0"   # ocean mechanism
    C_OCEAN2  = "#0d9ec4"   # ocean secondary
    C_BIOL    = "#1e8c5a"   # biological
    C_IMPACT  = "#d45500"   # ecosystem impact
    C_FOOD    = "#8b0000"   # food/human
    C_LAND    = "#7a4800"   # land impacts
    C_SIDE    = "#5a3570"   # side impacts

    # ── Title block ──
    ax.text(5, 19.4,
            "エルニーニョが生態系へ与える影響の連鎖",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#1a1a2e", zorder=10)
    ax.text(5, 18.9,
            "Fig. 3  El Niño Ecosystem Impact Cascade",
            ha="center", va="center", fontsize=10, color="#555566", zorder=10)

    # ── Main chain (center column) ──
    nodes = [
        # (x, y, width, height, text, color)
        (5, 18.0, 5.5, 0.75, "貿易風の弱化",                  C_DRIVER),
        (5, 16.9, 5.5, 0.75, "暖水の東進・中部〜東部太平洋の昇温",   C_OCEAN1),
        (5, 15.8, 5.5, 0.75, "アップウェリング（湧昇流）の弱化",     C_OCEAN1),
        (5, 14.7, 5.5, 0.75, "深層冷水・栄養塩の供給減少",          C_OCEAN2),
        (5, 13.5, 5.5, 0.80, "植物プランクトンの大幅減少",           C_BIOL),
        (5, 12.3, 5.5, 0.80, "動物プランクトン・小型魚類の減少",      C_BIOL),
        (5, 11.1, 5.5, 0.80, "食物連鎖全体の崩壊リスク",             C_IMPACT),
    ]

    for (x, y, w, h, text, color) in nodes:
        draw_box(ax, x, y, w, h, text, color, fontsize=10.5)

    # Vertical arrows in center chain
    chain_ys = [n[1] for n in nodes]
    chain_hs = [n[3] for n in nodes]
    for i in range(len(chain_ys) - 1):
        y_bottom = chain_ys[i] - chain_hs[i]/2
        y_top    = chain_ys[i+1] + chain_hs[i+1]/2
        arrow(ax, 5, y_bottom - 0.01, 5, y_top + 0.01, color="#334488", lw=2.5)

    # ── Branch: Marine impacts (right side) ──
    branch_y_start = 11.1  # from "食物連鎖崩壊リスク"

    # Coral bleaching
    draw_box(ax, 8.2, 10.2, 3.2, 0.75,
             "サンゴの白化・死滅\n（海洋熱波）",
             C_IMPACT, fontsize=9.5)
    arrow(ax, 6.75, 11.1, 7.55, 10.45, color="#cc4400")

    # Fisheries
    draw_box(ax, 8.2, 9.0, 3.2, 0.75,
             "漁業崩壊・\n魚類資源の激減",
             C_IMPACT, fontsize=9.5)
    arrow(ax, 6.75, 10.8, 7.55, 9.35, color="#cc4400")

    # Seabird & marine mammals
    draw_box(ax, 8.2, 7.8, 3.2, 0.75,
             "海鳥・海洋哺乳類\nの繁殖失敗・死亡",
             C_IMPACT, fontsize=9.5)
    arrow(ax, 6.75, 10.75, 7.55, 8.15, color="#cc4400")

    # ── Branch: Land/atmosphere impacts (left side) ──
    # Drought
    draw_box(ax, 1.8, 10.0, 3.2, 0.75,
             "干ばつ・降雨パターン\nの異常",
             C_LAND, fontsize=9.5)
    arrow(ax, 3.25, 11.1, 2.55, 10.35, color="#885500")

    # Wildfire
    draw_box(ax, 1.8, 8.8, 3.2, 0.75,
             "山火事リスクの激増\n（熱帯雨林・草原）",
             C_LAND, fontsize=9.5)
    arrow(ax, 1.8, 9.62, 1.8, 9.17, color="#885500")

    # Floods
    draw_box(ax, 1.8, 7.6, 3.2, 0.75,
             "洪水・土壌流失\n（別地域で豪雨）",
             C_LAND, fontsize=9.5)
    arrow(ax, 1.8, 8.42, 1.8, 7.97, color="#885500")

    # ── Lower convergence: food & human security ──
    draw_box(ax, 5, 6.5, 6.0, 0.85,
             "農業・食料安全保障への深刻な影響",
             C_FOOD, fontsize=11)
    # Arrows from marine and land branches
    arrow(ax, 8.2, 7.42, 6.1, 6.92, color="#881100", lw=2.2)
    arrow(ax, 1.8, 7.22, 3.9, 6.92, color="#881100", lw=2.2)
    arrow(ax, 5,  10.70, 5, 6.92, color="#881100", lw=2.2)

    # ── Ocean circulation global impact ──
    draw_box(ax, 5, 5.4, 7.0, 0.80,
             "海洋循環・熱循環・酸素循環・栄養循環の複合的撹乱",
             C_SIDE, fontsize=10)
    arrow(ax, 5, 6.07, 5, 5.80, color="#5a3570", lw=2.2)

    # ── Tipping point warning ──
    draw_box(ax, 5, 4.2, 8.0, 0.85,
             "⚠ 温暖化との相乗効果：生態系ティッピングポイントのリスク",
             "#5a0a0a", fontsize=10.5)
    arrow(ax, 5, 4.99, 5, 4.62, color="#880000", lw=2.5)

    # ── Final outcome ──
    draw_box(ax, 5, 2.9, 8.5, 0.90,
             "地球システムレベルでの生物多様性喪失・\n生産性崩壊・人類の食料・水資源危機",
             "#3a0000", fontsize=11)
    arrow(ax, 5, 3.75, 5, 3.35, color="#660000", lw=2.8)

    # ── Source note ──
    ax.text(5, 2.05, "出典：NOAA / WMO / IPCC AR6 / NASA / 気象庁（概念フローチャート）",
            ha="center", va="center", fontsize=7.5, color="#666688", zorder=10)

    # ── Category legend ──
    legend_items = [
        (C_DRIVER, "大気・物理的ドライバー"),
        (C_OCEAN1, "海洋メカニズム"),
        (C_BIOL,   "生物学的影響"),
        (C_IMPACT, "海洋生態系影響"),
        (C_LAND,   "陸域・大気影響"),
        (C_FOOD,   "食料・人間社会"),
        (C_SIDE,   "地球システム撹乱"),
    ]
    x_leg, y_leg = 0.4, 1.55
    for i, (col, label) in enumerate(legend_items):
        col_offset = (i % 4) * 2.4
        row_offset = (i // 4) * 0.55
        rect = FancyBboxPatch((x_leg + col_offset - 0.1, y_leg - row_offset - 0.22),
                              0.35, 0.35, boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor="white", lw=1, zorder=5)
        ax.add_patch(rect)
        ax.text(x_leg + col_offset + 0.35, y_leg - row_offset - 0.05,
                label, fontsize=8, color="#1a1a2e", va="center", zorder=6)

    fig.savefig("figures/el_nino_ecosystem_impact_flowchart.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Figure 3 saved.")


# ══════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
    print("All figures generated successfully.")
