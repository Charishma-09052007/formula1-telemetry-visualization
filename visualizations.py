"""
F1 Telemetry & Driver Performance Analysis - Phase 1 Visualizations
====================================================================
Q1: G-G Diagram (Friction Circle) - Combined Grip Analysis
Q2: Acceleration Decay Curve - Aerodynamic Profile
Q3: Throttle Variance Comparison - Pressure Analysis

MODIFIED: 
- Fixed Cache NotADirectoryError
- Implemented Q1 Performance Envelopes (Density Contours)
- Preserved original Q2/Q3 detailed analysis logic
"""

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
import warnings
import os

warnings.filterwarnings('ignore')

# Get base directory (script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIX: Ensure cache directory exists before enabling it
CACHE_DIR = os.path.join(BASE_DIR, 'ff1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2024
GP = 'Bahrain'

# Directory where raw telemetry CSVs are permanently saved
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def save_telemetry_csv(df: pd.DataFrame, filename: str) -> None:
    """Persist a telemetry DataFrame to CSV so the collected data survives git pushes."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        print(f"  ✓ Data saved: {path}")
    else:
        print(f"  ℹ Data already exists, skipping save: {path}")

print("=" * 60)
print("F1 Telemetry & Driver Performance Analysis")
print("=" * 60)


def load_session(year, gp, session_type):
    """Load a FastF1 session and return it."""
    print(f"\nLoading {year} {gp} {session_type}...")
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    print(f"  Loaded. Drivers: {len(session.drivers)}")
    return session


def get_merged_telemetry(session, driver_num):
    """
    Merge car_data and pos_data for a driver using FastF1's Telemetry.merge_channels().
    Returns a Telemetry DataFrame with Speed, Throttle, Brake, X, Y, Z, etc.
    """
    car = session.car_data[driver_num].copy()
    pos = session.pos_data[driver_num].copy()
    merged = car.merge_channels(pos)
    return merged


def compute_g_forces(tel_df):
    """
    Compute longitudinal and lateral G-forces from merged telemetry.
    Uses Speed and X/Y positional data.
    """
    df = tel_df.copy()

    # Time in seconds
    df['dt'] = df['SessionTime'].diff().dt.total_seconds()

    # Speed in m/s
    df['Speed_ms'] = df['Speed'] / 3.6

    # Longitudinal G: rate of change of speed
    df['dv'] = df['Speed_ms'].diff()
    df['G_long'] = (df['dv'] / df['dt']) / 9.81

    # Lateral G from positional data
    if 'X' in df.columns and 'Y' in df.columns:
        dx = df['X'].diff()
        dy = df['Y'].diff()
        heading = np.arctan2(dy, dx)
        dheading = heading.diff()
        # Wrap angle differences
        dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
        omega = dheading / df['dt']
        df['G_lat'] = (df['Speed_ms'] * omega) / 9.81
    else:
        df['G_lat'] = 0.0

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['G_long', 'G_lat', 'dt'])
    df = df[df['dt'] > 0]

    # Smooth
    if len(df) > 15:
        df['G_long'] = savgol_filter(df['G_long'], window_length=11, polyorder=3)
        df['G_lat'] = savgol_filter(df['G_lat'], window_length=11, polyorder=3)

    # Clip extreme outliers
    df = df[(df['G_long'].abs() < 6) & (df['G_lat'].abs() < 6)]

    return df


def draw_density_envelope(ax, gf, color, label):
    """
    Draws a contour line representing the 90th percentile density of the G-points.
    This shows the 'performance envelope' without being obscured by point overlap.
    """
    try:
        if len(gf) < 50: return
        x = gf['G_lat'].values
        y = gf['G_long'].values
        
        # Calculate kernel density
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Plot the contour at a specific level (0.15 threshold)
        ax.contour(xi, yi, zi.reshape(xi.shape), levels=[zi.max() * 0.15],
                   colors=color, linewidths=2, alpha=0.8, linestyles='solid')
        # Add a dummy line for the legend
        ax.plot([], [], color=color, label=f"{label} Envelope", linewidth=2)
    except Exception as e:
        print(f"  ! Could not draw envelope for {label}: {e}")


# ============================================================================
# QUESTION 1: G-G Diagram (Friction Circle) - Combined Grip Analysis
# Target User: Driver Performance Coach
# ============================================================================

def plot_gg_diagram(session_type, session_label, output_path):
    """
    Plot G-G Diagram (Friction Circle) comparing two drivers.
    Shows how efficiently each driver uses combined grip during corners.
    """
    print(f"\n--- Q1: G-G Diagram (Friction Circle) [{session_label}] ---")

    session = load_session(YEAR, GP, session_type)
    results = session.results

    # Compare P1 and P2
    d1_num = str(results.iloc[0]['DriverNumber'])
    d2_num = str(results.iloc[1]['DriverNumber'])
    d1_abbr = results.iloc[0]['Abbreviation']
    d2_abbr = results.iloc[1]['Abbreviation']
    d1_name = results.iloc[0]['FullName']
    d2_name = results.iloc[1]['FullName']

    print(f"  Comparing: {d1_name} vs {d2_name}")

    # Get merged telemetry
    tel1 = get_merged_telemetry(session, d1_num)
    tel2 = get_merged_telemetry(session, d2_num)

    # Filter to driving only
    tel1 = tel1[tel1['Speed'] > 50]
    tel2 = tel2[tel2['Speed'] > 50]

    # Sample a mid-race window
    total_time1 = tel1['SessionTime'].max().total_seconds()
    mid1 = total_time1 / 2
    mask1 = (tel1['SessionTime'].dt.total_seconds() > mid1 - 300) & \
            (tel1['SessionTime'].dt.total_seconds() < mid1 + 300)
    tel1_sample = tel1[mask1]

    total_time2 = tel2['SessionTime'].max().total_seconds()
    mid2 = total_time2 / 2
    mask2 = (tel2['SessionTime'].dt.total_seconds() > mid2 - 300) & \
            (tel2['SessionTime'].dt.total_seconds() < mid2 + 300)
    tel2_sample = tel2[mask2]

    gf1 = compute_g_forces(tel1_sample)
    gf2 = compute_g_forces(tel2_sample)

    print(f"  Data points: {d1_abbr}={len(gf1)}, {d2_abbr}={len(gf2)}")

    # Persist collected telemetry to CSV
    safe_label = session_label.replace(' ', '_')
    save_telemetry_csv(gf1, f"Q1_gforces_{safe_label}_{d1_abbr}.csv")
    save_telemetry_csv(gf2, f"Q1_gforces_{safe_label}_{d2_abbr}.csv")

    # Force red/green for driver distinction
    color1 = '#E10600'
    color2 = '#00D000'

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#1a1a2e')

    fig.suptitle(
        f"G-G Diagram (Friction Circle) — Combined Grip Analysis\n"
        f"{YEAR} {GP} GP {session_label} • Performance Envelope Comparison",
        fontsize=14, fontweight='bold', color='white', y=1.02
    )

    max_g1 = max(gf1['G_long'].abs().quantile(0.98), gf1['G_lat'].abs().quantile(0.98))
    max_g2 = max(gf2['G_long'].abs().quantile(0.98), gf2['G_lat'].abs().quantile(0.98))
    max_g_both = max(max_g1, max_g2) * 1.15

    for i, (ax, gf, color, name, abbr) in enumerate([
        (axes[0], gf1, color1, d1_name, d1_abbr),
        (axes[1], gf2, color2, d2_name, d2_abbr),
    ]):
        ax.set_facecolor('#16213e')
        ax.scatter(gf['G_lat'], gf['G_long'], s=2, alpha=0.35, c=color, label=abbr)
        # Draw envelope in white for individual plots
        draw_density_envelope(ax, gf, 'white', abbr)
        
        circle = Circle((0, 0), max_g_both * 0.85, fill=False, color='gray',
                        linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)
        ax.set_xlim(-max_g_both, max_g_both)
        ax.set_ylim(-max_g_both, max_g_both)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Lateral G (Turning)', fontsize=11, color='white')
        ax.set_ylabel('Longitudinal G (Braking ↓ / Accel ↑)', fontsize=11, color='white')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold', color=color)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_color('gray')

    ax = axes[2]
    ax.set_facecolor('#16213e')
    # Use higher transparency for points in overlay to focus on envelopes
    ax.scatter(gf1['G_lat'], gf1['G_long'], s=2, alpha=0.1, c=color1)
    ax.scatter(gf2['G_lat'], gf2['G_long'], s=2, alpha=0.1, c=color2)
    
    # Draw Envelopes as the primary comparison tool
    draw_density_envelope(ax, gf1, color1, d1_abbr)
    draw_density_envelope(ax, gf2, color2, d2_abbr)
    
    circle3 = Circle((0, 0), max_g_both * 0.85, fill=False, color='white',
                     linestyle='--', linewidth=1.5, alpha=0.4, label='Friction Limit')
    ax.add_patch(circle3)
    ax.set_xlim(-max_g_both, max_g_both)
    ax.set_ylim(-max_g_both, max_g_both)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Lateral G (Turning)', fontsize=11, color='white')
    ax.set_ylabel('Longitudinal G (Braking ↓ / Accel ↑)', fontsize=11, color='white')
    ax.set_title('Overlay: Teammate Envelope', fontsize=12, fontweight='bold', color='white')
    ax.legend(fontsize=9, loc='upper right', facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color('gray')

    for a in axes:
        a.annotate('BRAKING', xy=(0, -max_g_both * 0.75), fontsize=8,
                   ha='center', color='#aaa', style='italic')
        a.annotate('ACCEL', xy=(0, max_g_both * 0.75), fontsize=8,
                   ha='center', color='#aaa', style='italic')
        a.annotate('LEFT', xy=(-max_g_both * 0.75, 0), fontsize=8,
                   ha='center', va='center', color='#aaa', style='italic', rotation=90)
        a.annotate('RIGHT', xy=(max_g_both * 0.75, 0), fontsize=8,
                   ha='center', va='center', color='#aaa', style='italic', rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# QUESTION 2: Acceleration Decay Curve - Aerodynamic Profile
# Target User: Aerodynamicist / Race Strategist
# ============================================================================

def plot_acceleration_decay():
    """
    Plot Acceleration Decay Curve comparing cars from different teams.
    Isolates the 'Aerodynamic Wall' to distinguish power vs drag issues.
    """
    print("\n--- Q2: Acceleration Decay Curve ---")

    session = load_session(YEAR, GP, 'R')
    results = session.results

    # Pick top driver from 4 different teams
    teams_seen = set()
    drivers_info = []
    for _, row in results.iterrows():
        team = row['TeamName']
        if team not in teams_seen and len(drivers_info) < 4:
            teams_seen.add(team)
            drivers_info.append({
                'num': str(row['DriverNumber']),
                'abbr': row['Abbreviation'],
                'name': row['FullName'],
                'team': team,
                'color': '#' + str(row['TeamColor']) if pd.notna(row['TeamColor']) else None
            })

    fallback_colors = ['#3671C6', '#E80020', '#27F4D2', '#FF8000']
    print(f"  Teams: {[d['team'] for d in drivers_info]}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        f"Acceleration Decay Curve — Aerodynamic Profile Analysis\n"
        f"{YEAR} {GP} GP Race • Full Throttle Segments Only",
        fontsize=14, fontweight='bold', color='white', y=1.02
    )

    all_curves = []
    all_binned = []

    for idx, d in enumerate(drivers_info):
        car = session.car_data[d['num']].copy()
        car = car[car['Speed'] > 30]

        car['dt'] = car['SessionTime'].diff().dt.total_seconds()
        car['Speed_ms'] = car['Speed'] / 3.6
        car['dv'] = car['Speed_ms'].diff()
        car['G_long'] = (car['dv'] / car['dt']) / 9.81

        mask = (car['Throttle'] >= 95) & (car['G_long'] > 0) & (car['dt'] > 0) & (car['dt'] < 1)
        filt = car[mask].copy()
        filt = filt.replace([np.inf, -np.inf], np.nan).dropna(subset=['G_long', 'Speed'])
        filt = filt[filt['G_long'] < 3]

        filt['SpeedBin'] = (filt['Speed'] // 10) * 10
        binned = filt.groupby('SpeedBin').agg(
            G_mean=('G_long', 'median'),
            G_std=('G_long', 'std'),
            count=('G_long', 'count')
        ).reset_index()
        binned = binned[binned['count'] >= 10]
        binned = binned[(binned['SpeedBin'] >= 80) & (binned['SpeedBin'] <= 330)]

        color = d['color'] or fallback_colors[idx]
        all_curves.append({'binned': binned, 'color': color,
                           'label': f"{d['abbr']} ({d['team']})"})
        all_binned.append((d['abbr'], binned))

    # Persist Q2 binned acceleration data
    for abbr, b in all_binned:
        b_export = b.copy()
        b_export.insert(0, 'Driver', abbr)
        save_telemetry_csv(b_export, f"Q2_accel_decay_{abbr}.csv")

    ax = axes[0]
    ax.set_facecolor('#16213e')
    for curve in all_curves:
        b = curve['binned']
        ax.plot(b['SpeedBin'], b['G_mean'], '-o', color=curve['color'],
                label=curve['label'], linewidth=2.5, markersize=5)
        ax.fill_between(b['SpeedBin'],
                        b['G_mean'] - b['G_std'] * 0.5,
                        b['G_mean'] + b['G_std'] * 0.5,
                        color=curve['color'], alpha=0.1)

    ax.axhline(0, color='white', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel('Speed (km/h)', fontsize=12, color='white')
    ax.set_ylabel('Longitudinal Acceleration (G)', fontsize=12, color='white')
    ax.set_title('Acceleration vs Speed — Full Throttle', fontsize=12,
                 fontweight='bold', color='white')
    ax.legend(fontsize=9, loc='upper right', facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_ylim(bottom=-0.05)
    for spine in ax.spines.values():
        spine.set_color('gray')

    ax.annotate('← Drag-Limited Region\n   (Aero Wall)',
                xy=(280, 0.2), fontsize=9, color='#ffcc00', style='italic', ha='center')
    ax.annotate('← Traction-Limited',
                xy=(120, 0.8), fontsize=9, color='#aaa', style='italic', ha='center')

    ax = axes[1]
    ax.set_facecolor('#16213e')
    for curve in all_curves:
        b = curve['binned']
        if len(b) > 1:
            max_g = b['G_mean'].max()
            if max_g > 0:
                b_norm = b['G_mean'] / max_g * 100
                ax.plot(b['SpeedBin'], b_norm, '-s', color=curve['color'],
                        label=curve['label'], linewidth=2.5, markersize=5)

    ax.axhline(50, color='gray', linewidth=1, linestyle=':', alpha=0.6)
    ax.set_xlabel('Speed (km/h)', fontsize=12, color='white')
    ax.set_ylabel('Normalized Acceleration (%)', fontsize=12, color='white')
    ax.set_title('Normalized Decay — Aero Efficiency Comparison', fontsize=12,
                 fontweight='bold', color='white')
    ax.legend(fontsize=9, loc='upper right', facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    ax.set_ylim(0, 110)
    for spine in ax.spines.values():
        spine.set_color('gray')

    ax.annotate('Low Drag Profile\n(maintains acceleration)',
                xy=(290, 80), fontsize=9, color='#00ff88', style='italic', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e', edgecolor='#00ff88', alpha=0.8))
    ax.annotate('High Drag Profile\n(rapid decay)',
                xy=(290, 20), fontsize=9, color='#ff6666', style='italic', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e', edgecolor='#ff6666', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, 'Q2_Acceleration_Decay_Curve.png')
    plt.savefig(output_path,
                dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("  ✓ Saved: Q2_Acceleration_Decay_Curve.png")


# ============================================================================
# QUESTION 3: Throttle Variance Comparison - Pressure Analysis
# Target User: Performance Psychologist / Team Principal
# ============================================================================

def plot_throttle_variance():
    """
    Compare throttle smoothness: free air vs under pressure (being chased).
    Uses car_data directly to extract throttle traces and segments the race.
    """
    print("\n--- Q3: Throttle Variance Comparison ---")

    session = load_session(YEAR, GP, 'R')
    results = session.results

    target_num = str(results.iloc[0]['DriverNumber'])
    target_name = results.iloc[0]['FullName']
    target_abbr = results.iloc[0]['Abbreviation']

    print(f"  Analyzing: {target_name} ({target_abbr})")

    car = session.car_data[target_num].copy()
    car = car[car['Speed'] > 0]

    total_seconds = car['SessionTime'].max().total_seconds()
    approx_lap_time = 95
    n_laps_approx = int(total_seconds / approx_lap_time)

    segments = []
    session_time = car['SessionTime'].dt.total_seconds().values

    for i in range(n_laps_approx):
        t_start = i * approx_lap_time
        t_end = (i + 1) * approx_lap_time
        mask = (session_time >= t_start) & (session_time < t_end)
        seg_data = car[mask]
        if len(seg_data) > 50:
            segments.append({
                'data': seg_data,
                'lap_num': i + 1,
                'throttle': seg_data['Throttle'].values,
                'speed': seg_data['Speed'].values,
                'time': seg_data['SessionTime'].dt.total_seconds().values
            })

    print(f"  Total segments (pseudo-laps): {len(segments)}")

    # Persist Q3 raw throttle car data
    car_export = car.copy()
    save_telemetry_csv(car_export, f"Q3_throttle_{target_abbr}.csv")

    n_segs = len(segments)
    third = max(n_segs // 3, 3)
    pressure_segs = segments[1:third + 1]
    free_air_segs = segments[-third - 1:-1]

    def compute_throttle_metrics(segs):
        variances = []
        jitters = []
        traces = []
        for seg in segs:
            throttle = seg['throttle']
            if len(throttle) < 20:
                continue
            roll_std = pd.Series(throttle).rolling(15).std().dropna().values
            variances.append(np.mean(roll_std))
            jitter = np.std(np.diff(throttle))
            jitters.append(jitter)
            traces.append(seg)
        return traces, variances, jitters

    p_traces, p_vars, p_jitters = compute_throttle_metrics(pressure_segs)
    f_traces, f_vars, f_jitters = compute_throttle_metrics(free_air_segs)

    print(f"  Pressure segments: {len(p_traces)}, Free air segments: {len(f_traces)}")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        f"Throttle Smoothness Under Pressure — Nervousness Detector\n"
        f"{YEAR} {GP} GP Race • {target_name} ({target_abbr})",
        fontsize=14, fontweight='bold', color='white', y=1.02
    )

    p_color = '#E10600'
    f_color = '#00D2BE'

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#16213e')
    for trace in p_traces[:5]:
        norm_dist = np.linspace(0, 100, len(trace['throttle']))
        ax1.plot(norm_dist, trace['throttle'], alpha=0.5, color=p_color,
                 linewidth=0.8, label=f"Seg {trace['lap_num']}")
    ax1.set_xlabel('Normalized Segment Distance (%)', fontsize=10, color='white')
    ax1.set_ylabel('Throttle (%)', fontsize=10, color='white')
    ax1.set_title('Scenario B: Under Pressure\n(Early Race — Traffic)',
                  fontsize=11, fontweight='bold', color=p_color)
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=7, loc='lower right', facecolor='#16213e', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    for trace in f_traces[:5]:
        norm_dist = np.linspace(0, 100, len(trace['throttle']))
        ax2.plot(norm_dist, trace['throttle'], alpha=0.5, color=f_color,
                 linewidth=0.8, label=f"Seg {trace['lap_num']}")
    ax2.set_xlabel('Normalized Segment Distance (%)', fontsize=10, color='white')
    ax2.set_ylabel('Throttle (%)', fontsize=10, color='white')
    ax2.set_title('Scenario A: Free Air\n(Late Race — Clean Air)',
                  fontsize=11, fontweight='bold', color=f_color)
    ax2.set_ylim(-5, 105)
    ax2.legend(fontsize=7, loc='lower right', facecolor='#16213e', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#16213e')
    if p_traces and f_traces:
        norm_p = np.linspace(0, 100, len(p_traces[0]['throttle']))
        norm_f = np.linspace(0, 100, len(f_traces[0]['throttle']))
        ax3.plot(norm_p, p_traces[0]['throttle'], color=p_color, linewidth=1.2,
                 alpha=0.9, label=f"Under Pressure (Seg {p_traces[0]['lap_num']})")
        ax3.plot(norm_f, f_traces[0]['throttle'], color=f_color, linewidth=1.2,
                 alpha=0.9, label=f"Free Air (Seg {f_traces[0]['lap_num']})")
    ax3.set_xlabel('Normalized Segment Distance (%)', fontsize=10, color='white')
    ax3.set_ylabel('Throttle (%)', fontsize=10, color='white')
    ax3.set_title('Direct Overlay Comparison', fontsize=11, fontweight='bold', color='white')
    ax3.set_ylim(-5, 105)
    ax3.legend(fontsize=8, loc='lower right', facecolor='#16213e', labelcolor='white')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.15, color='white')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#16213e')
    categories = ['Under Pressure\n(Scenario B)', 'Free Air\n(Scenario A)']
    mean_vars = [np.mean(p_vars) if p_vars else 0,
                 np.mean(f_vars) if f_vars else 0]
    std_vars = [np.std(p_vars) if len(p_vars) > 1 else 0,
                np.std(f_vars) if len(f_vars) > 1 else 0]
    bars = ax4.bar(categories, mean_vars, yerr=std_vars, capsize=8,
                   color=[p_color, f_color], edgecolor='white', linewidth=0.8, alpha=0.85)
    ax4.set_ylabel('Mean Rolling Throttle Std Dev', fontsize=10, color='white')
    ax4.set_title('Throttle Variance Comparison', fontsize=11, fontweight='bold', color='white')
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.15, color='white', axis='y')
    for bar, val in zip(bars, mean_vars):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', fontsize=11, fontweight='bold', color='white')
    for spine in ax4.spines.values():
        spine.set_color('gray')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#16213e')
    box_data = []
    box_labels = []
    if p_jitters:
        box_data.append(p_jitters)
        box_labels.append('Under Pressure')
    if f_jitters:
        box_data.append(f_jitters)
        box_labels.append('Free Air')
    if box_data:
        bp = ax5.boxplot(box_data, labels=box_labels, patch_artist=True,
                         widths=0.5, notch=True,
                         boxprops=dict(linewidth=1.5),
                         whiskerprops=dict(color='white'),
                         capprops=dict(color='white'),
                         medianprops=dict(color='yellow', linewidth=2),
                         flierprops=dict(markeredgecolor='white'))
        colors_box = [p_color, f_color][:len(box_data)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax5.set_ylabel('Throttle Derivative Std Dev (Jitter)', fontsize=10, color='white')
    ax5.set_title('High-Frequency Jitter Distribution', fontsize=11,
                  fontweight='bold', color='white')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.15, color='white', axis='y')
    for spine in ax5.spines.values():
        spine.set_color('gray')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#16213e')
    if p_traces and f_traces:
        lap_nums_p = [t['lap_num'] for t in p_traces]
        lap_nums_f = [t['lap_num'] for t in f_traces]

        ax6.scatter(lap_nums_p, p_vars[:len(lap_nums_p)],
                    c=p_color, s=60, label='Under Pressure', edgecolors='white',
                    linewidth=0.5, zorder=3)
        ax6.scatter(lap_nums_f, f_vars[:len(lap_nums_f)],
                    c=f_color, s=60, label='Free Air', edgecolors='white',
                    linewidth=0.5, zorder=3)

        if len(lap_nums_p) > 1:
            z = np.polyfit(lap_nums_p, p_vars[:len(lap_nums_p)], 1)
            p = np.poly1d(z)
            ax6.plot(lap_nums_p, p(lap_nums_p), '--', color=p_color, linewidth=1.5, alpha=0.7)
        if len(lap_nums_f) > 1:
            z = np.polyfit(lap_nums_f, f_vars[:len(lap_nums_f)], 1)
            p = np.poly1d(z)
            ax6.plot(lap_nums_f, p(lap_nums_f), '--', color=f_color, linewidth=1.5, alpha=0.7)

    ax6.set_xlabel('Segment Number (≈Lap)', fontsize=10, color='white')
    ax6.set_ylabel('Throttle Variance (Std Dev)', fontsize=10, color='white')
    ax6.set_title('Variance Trend Across Race', fontsize=11, fontweight='bold', color='white')
    ax6.legend(fontsize=9, facecolor='#16213e', labelcolor='white')
    ax6.tick_params(colors='white')
    ax6.grid(True, alpha=0.15, color='white')
    for spine in ax6.spines.values():
        spine.set_color('gray')

    output_path = os.path.join(BASE_DIR, 'Q3_Throttle_Variance_Comparison.png')
    plt.savefig(output_path,
                dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {output_path}")


if __name__ == '__main__':
    race_output = os.path.join(BASE_DIR, 'Q1_GG_Diagram_Friction_Circle.png')
    qual_output = os.path.join(BASE_DIR, 'Q1_GG_Diagram_Friction_Circle_Qualifying.png')
    plot_gg_diagram('R', 'Race', race_output)
    plot_gg_diagram('Q', 'Qualifying', qual_output)
    plot_acceleration_decay()
    plot_throttle_variance()
    print("\n" + "=" * 60)
    print("All three visualizations generated successfully!")
    print("=" * 60)
    print(f"Files in {BASE_DIR}/")
    print("  Q1_GG_Diagram_Friction_Circle.png")
    print("  Q1_GG_Diagram_Friction_Circle_Qualifying.png")
    print("  Q2_Acceleration_Decay_Curve.png")
    print("  Q3_Throttle_Variance_Comparison.png")