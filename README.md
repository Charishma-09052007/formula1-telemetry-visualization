# Formula 1 Telemetry Visualization

## Overview

This project analyzes and visualizes Formula 1 telemetry data to study vehicle performance characteristics during qualifying and race sessions. The analysis focuses on driver behavior, acceleration patterns, throttle application, and lateral/longitudinal vehicle dynamics using real telemetry datasets.

Interactive and static visualizations are generated to compare drivers and teams across multiple performance metrics.

## Objectives

* Analyze Formula 1 telemetry datasets
* Compare driver and team performance characteristics
* Visualize vehicle dynamics using telemetry measurements
* Identify differences in acceleration, braking, and throttle behavior
* Present insights through graphical dashboards and plots

## Analyses Performed

### 1. G-G Diagram Analysis

A G-G (friction circle) diagram visualizes the relationship between lateral and longitudinal acceleration.

The analysis compares:

* Race vs Qualifying performance
* Driver-specific telemetry
* Team-level performance characteristics

Generated Visualizations:

* Friction Circle
* G-G Diagram (Race)
* G-G Diagram (Qualifying)
* Team Comparison Plots

### 2. Acceleration Decay Analysis

Investigates how acceleration changes at increasing speeds and compares different drivers.

Metrics analyzed:

* Longitudinal acceleration
* Speed growth patterns
* Acceleration decay curves

Generated Visualization:

* Acceleration Decay Curve

### 3. Throttle Variance Analysis

Examines throttle application consistency and driver control characteristics.

Metrics analyzed:

* Throttle position
* Variance in throttle input
* Driver behavior comparison

Generated Visualization:

* Throttle Variance Comparison

## Project Structure

```text
formula1-telemetry-visualization/
│
├── data/
│   ├── G-force datasets
│   ├── Acceleration datasets
│   └── Throttle datasets
│
├── images/
│   ├── Q1_GG_Diagram_Friction_Circle.png
│   ├── Q2_Acceleration_Decay_Curve.png
│   └── Q3_Throttle_Variance_Comparison.png
│
├── prepare_data.py
├── visualizations.py
├── gg_qualifying_teams.py
├── dashboard_data.json
├── index.html
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* FastF1
* NumPy
* Pandas
* SciPy
* Matplotlib
* HTML
* JavaScript

## Key Visualizations

* G-G Diagram / Friction Circle Analysis
* Team Performance Comparison
* Acceleration Decay Curves
* Throttle Variance Analysis

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis scripts:

```bash
python prepare_data.py
python visualizations.py
```

Open the dashboard:

```bash
index.html
```

## Results

The project demonstrates how telemetry data can be used to analyze:

* Driver consistency
* Vehicle dynamics
* Acceleration characteristics
* Team performance differences
* Throttle control behavior

The generated visualizations provide an intuitive view of Formula 1 performance metrics and racing dynamics.
