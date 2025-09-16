# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6 (Parameter Recovery for Varying Network Sizes)

These commands run the simulations and produce Figure 2 and Figure 3.

```bash
>>> cd parameter_recovery/
>>> python simulation.py
>>> python process.py
>>> python plot_node.py
>>> python plot_time.py
```

### Section 7 (Forecasting International Conflicts)

To produce the figures and compute the goodnees-of-fit metrics for the real data application, 
you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook polecat_application/POLECAT\ Application.ipynb
```

### Section C.1 (Visualization of Latent Trajectories)

To produce Figure S.1 in the supplement, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Parameter\ Recovery\ Visualizations.ipynb
```

### Section C.2 (Forecasting Performance)

These commands run the simulations and produce Figure S.2

```bash
>>> cd forecast_comparison
>>> python simulation.py
>>> python process.py
>>> python plot_forecast.py
```

To produce Figure S.3, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Forecast\ Visualizations.ipynb
```
