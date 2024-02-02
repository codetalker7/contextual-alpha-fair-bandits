python3 -m config --ROWS=10000 --SEED=1 --ALPHA=0.9 --SMALLREWARD=0.001 --USETIMESTAMPS --FREQUENCY=5000 --FAIRCBFAIRNESS=0.9 --PLOTREGRET
python3 -m data_exploration
python3 -m offline_optimal
python3 -m ml25m
python3 -m ml25m_bandit
