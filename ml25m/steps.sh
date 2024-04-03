python3 -m config --ROWS=50000 --SEED=1 --ALPHA=0.9 --SMALLREWARD=0.001 --USETIMESTAMPS --FREQUENCY=1000 --FREQUENCY_MAX=1000 --HIGHFREQUENCY --FAIRCBFAIRNESS=0.98 --PLOTREGRET --NUMNUS=50 --NUMALPHAS=50 --VARYING_NU_ROUNDS=10
python3 -m data_exploration
python3 -m offline_optimal_optimized
python3 -m ml25m
python3 -m ml25m_plots
python3 -m ml25m_bandit
python3 -m ml25m_bandit_plots
python3 -m dependence_on_alpha
python3 -m dependence_on_alpha_plots
python3 -m dependence_on_nu
python3 -m dependence_on_nu_plots