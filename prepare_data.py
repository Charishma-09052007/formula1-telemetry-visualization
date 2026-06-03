"""
Prepare CSV data from visualizations.py output into a single JSON file
for the D3 interactive dashboard.
"""
import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_PATH = os.path.join(BASE_DIR, 'dashboard_data.json')


def load_q1_gforces(filepath, max_points=2000):
    """Load G-force CSV; downsample if needed for browser performance."""
    df = pd.read_csv(filepath)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    return df[['G_lat', 'G_long']].round(4).to_dict(orient='records')


def load_q2_accel_decay(filepath):
    """Load Q2 binned acceleration decay CSV."""
    df = pd.read_csv(filepath)
    return df.to_dict(orient='records')


def compute_q3_metrics():
    """
    Replicate the Q3 throttle variance computation from visualizations.py
    to produce data for graphs 4 (bar chart) and 6 (scatter trend).
    """
    csv_path = os.path.join(DATA_DIR, 'Q3_throttle_VER.csv')
    car = pd.read_csv(csv_path)
    car = car[car['Speed'] > 0].copy()

    # Parse SessionTime to seconds
    car['SessionSeconds'] = car['SessionTime'].apply(parse_timedelta_to_seconds)
    car = car.dropna(subset=['SessionSeconds'])

    total_seconds = car['SessionSeconds'].max()
    approx_lap_time = 95
    n_laps_approx = int(total_seconds / approx_lap_time)

    segments = []
    for i in range(n_laps_approx):
        t_start = i * approx_lap_time
        t_end = (i + 1) * approx_lap_time
        mask = (car['SessionSeconds'] >= t_start) & (car['SessionSeconds'] < t_end)
        seg_data = car[mask]
        if len(seg_data) > 50:
            segments.append({
                'lap_num': i + 1,
                'throttle': seg_data['Throttle'].values,
            })

    n_segs = len(segments)
    third = max(n_segs // 3, 3)
    pressure_segs = segments[1:third + 1]
    free_air_segs = segments[-third - 1:-1]

    def compute_metrics(segs):
        variances = []
        results = []
        for seg in segs:
            throttle = seg['throttle']
            if len(throttle) < 20:
                continue
            roll_std = pd.Series(throttle).rolling(15).std().dropna().values
            var_val = float(np.mean(roll_std))
            variances.append(var_val)
            results.append({'lap_num': seg['lap_num'], 'variance': round(var_val, 4)})
        return results, variances

    p_results, p_vars = compute_metrics(pressure_segs)
    f_results, f_vars = compute_metrics(free_air_segs)

    # Graph 4: Bar chart data
    bar_data = {
        'pressure': {
            'mean': round(float(np.mean(p_vars)), 4) if p_vars else 0,
            'std': round(float(np.std(p_vars)), 4) if len(p_vars) > 1 else 0,
        },
        'free_air': {
            'mean': round(float(np.mean(f_vars)), 4) if f_vars else 0,
            'std': round(float(np.std(f_vars)), 4) if len(f_vars) > 1 else 0,
        }
    }

    # Graph 6: Scatter + trend data
    def trend_line(results):
        if len(results) < 2:
            return []
        x = [r['lap_num'] for r in results]
        y = [r['variance'] for r in results]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        return [{'lap_num': xi, 'trend': round(float(p(xi)), 4)} for xi in x]

    scatter_data = {
        'pressure': p_results,
        'free_air': f_results,
        'pressure_trend': trend_line(p_results),
        'free_air_trend': trend_line(f_results),
    }

    return bar_data, scatter_data


def parse_timedelta_to_seconds(td_str):
    """Parse '0 days HH:MM:SS.ffffff' to total seconds."""
    try:
        td = pd.Timedelta(td_str)
        return td.total_seconds()
    except Exception:
        return None


def main():
    data = {}

    # --- Q1: G-G Diagrams ---
    # 5 graphs: Race (VER vs PER), Qualifying (VER vs LEC), Qualifying teams (RB, Merc, McLaren)
    q1 = {}

    # Race: VER vs PER
    q1['race'] = {
        'title': 'Race — VER vs PER',
        'd1': {'abbr': 'VER', 'name': 'Max Verstappen', 'color': '#E10600',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Race_VER.csv'))},
        'd2': {'abbr': 'PER', 'name': 'Sergio Perez', 'color': '#00D000',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Race_PER.csv'))},
    }

    # Qualifying: VER vs LEC
    q1['qualifying'] = {
        'title': 'Qualifying — VER vs LEC',
        'd1': {'abbr': 'VER', 'name': 'Max Verstappen', 'color': '#E10600',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_VER.csv'))},
        'd2': {'abbr': 'LEC', 'name': 'Charles Leclerc', 'color': '#00D000',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_LEC.csv'))},
    }

    # Qualifying teams
    q1['red_bull'] = {
        'title': 'Qualifying — Red Bull',
        'd1': {'abbr': 'VER', 'name': 'Max Verstappen', 'color': '#E10600',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_Red_Bull_VER.csv'))},
        'd2': {'abbr': 'PER', 'name': 'Sergio Perez', 'color': '#00D000',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_Red_Bull_PER.csv'))},
    }
    q1['mercedes'] = {
        'title': 'Qualifying — Mercedes',
        'd1': {'abbr': 'HAM', 'name': 'Lewis Hamilton', 'color': '#E10600',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_Mercedes_HAM.csv'))},
        'd2': {'abbr': 'RUS', 'name': 'George Russell', 'color': '#00D000',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_Mercedes_RUS.csv'))},
    }
    q1['mclaren'] = {
        'title': 'Qualifying — McLaren',
        'd1': {'abbr': 'NOR', 'name': 'Lando Norris', 'color': '#E10600',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_McLaren_NOR.csv'))},
        'd2': {'abbr': 'PIA', 'name': 'Oscar Piastri', 'color': '#00D000',
               'data': load_q1_gforces(os.path.join(DATA_DIR, 'Q1_gforces_Qualifying_McLaren_PIA.csv'))},
    }
    data['q1'] = q1

    # --- Q2: Acceleration Decay ---
    q2_teams = []
    for fname in ['Q2_accel_decay_VER.csv', 'Q2_accel_decay_SAI.csv',
                   'Q2_accel_decay_NOR.csv', 'Q2_accel_decay_RUS.csv']:
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        abbr = df['Driver'].iloc[0]
        team_map = {'VER': 'Red Bull Racing', 'SAI': 'Ferrari', 'NOR': 'McLaren', 'RUS': 'Mercedes'}
        color_map = {'VER': '#3671C6', 'SAI': '#E80020', 'NOR': '#FF8000', 'RUS': '#27F4D2'}
        rows = df[['SpeedBin', 'G_mean', 'G_std', 'count']].to_dict(orient='records')
        # Compute normalized values
        max_g = df['G_mean'].max()
        for r in rows:
            r['G_norm'] = round(r['G_mean'] / max_g * 100, 2) if max_g > 0 else 0
        q2_teams.append({
            'abbr': abbr,
            'team': team_map.get(abbr, abbr),
            'color': color_map.get(abbr, '#888888'),
            'label': f"{abbr} ({team_map.get(abbr, abbr)})",
            'data': rows
        })
    data['q2'] = q2_teams

    # --- Q3: Throttle Variance (graphs 4 and 6 only) ---
    bar_data, scatter_data = compute_q3_metrics()
    data['q3'] = {
        'driver': 'Max Verstappen (VER)',
        'bar': bar_data,
        'scatter': scatter_data,
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(data, f)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"✓ Dashboard data written to {OUT_PATH} ({size_mb:.2f} MB)")

    # Embed JSON into dashboard.html so it works as a standalone file
    html_path = os.path.join(BASE_DIR, 'dashboard.html')
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html = f.read()
        placeholder = '<!-- DATA_PLACEHOLDER -->'
        if placeholder in html:
            data_script = f'<script>\nvar DASHBOARD_DATA = {json.dumps(data)};\n</script>'
            html = html.replace(placeholder, data_script)
            with open(html_path, 'w') as f:
                f.write(html)
            print(f"✓ Data embedded into {html_path}")
        else:
            print(f"⚠ Placeholder not found in {html_path}, data not embedded")
    else:
        print(f"⚠ {html_path} not found, skipping embed")


if __name__ == '__main__':
    main()
