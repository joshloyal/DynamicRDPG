# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6 and Section F.2 (Parameter Recovery for Varying Network Sizes)

These commands run the simulations and produce Figure 2 and Figure 3.

```bash
>>> cd parameter_recovery/
>>> python simulation.py
>>> python process.py
>>> python plot_node.py
>>> python plot_time.py
```

To produce Figures S.1-S.5 in the supplement, run the following commands

```bash
>>> python plot_node_gof.py
>>> python plot_time_gof.py
```

### Section 7 (Forecasting International Conflicts)

To produce the figures and compute the goodnees-of-fit metrics for the real data application, 
you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook polecat_application/POLECAT\ Application.ipynb
```

To run the backtesting procedure for the GB-DASE parameter tuning, you will need to run these commands to produce Table S.6.

```bash
>>> cd polecat_application/
>>> python backtesting.py
```

### Section F.1 (Visualization of Latent Trajectories)

To produce Figure S.1 in the supplement, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Parameter\ Recovery\ Visualizations.ipynb
```

### Section F.3 (Forecasting Performance)

These commands run the simulations and produce Figures S.7-S.9

```bash
>>> cd forecast_comparison/
>>> python simulation.py
>>> python process.py
>>> python plot_forecast.py
>>> python plot_forecast_metric.py
```

To produce Figure S.6, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Forecast\ Visualizations.ipynb
```

### Section F.4 (Parameter Recovery for Time-Varying Edge-Densities)

These commands run the simulations and produce Figures S.10-S.12

```bash
>>> cd parameter_recovery_time_varying_density/
>>> python simulation.py
>>> python process.py
>>> python plots.py
```

### Section F.5 (Parameter Recovery for Weighted Dynamic Networks)

These commands run the simulations and produce Figures S.13-S.16

```bash
>>> cd parameter_recovery_weighted/
>>> python simulation.py
>>> python process.py
>>> python plot_node.py
>>> python plot_time.py
```

## Section F.6 (Parameter Selection)

These commands run the simulations and produce the numbers in Table S.1

```bash
>>> cd parameter_selection_weighted/
>>> python simulation.py
>>> python process_comp.py
```

These commands run the simulations and produce the numbers in Table S.2

```bash
>> cd parameter_selection/
>>> python simulation.py
>>> python process_comp.py
```

## Section F.7 (Dimension Misspecification)

These commands run the simulations and produce the numbers in Tables S.4-S.5

```bash
>>> cd dimension_misspecification/
>>> python simulation.py
>>> python process.py
>>> python compare.py
```
