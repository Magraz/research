### To Install
Project requires Python 3.12

Run the following commands inside `src/vmas_salps` after cloning:
1. `git submodule init`
2. `git submodule update`
3. `cd VectorizedMultiAgentSimulator`
4. `pip install -e .`
7. `cd ../src`
8. `pip install -r requirements.txt`

### To Run Experiment

Run the following command for running experiments in different modalities:

- `python3 src/run_trial.py --batch=salp_local_4a --name=mlp --algorithm=ppo --environment=salp --trial_id=0`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel bash src/run_trial.py salp_local_4a mlp ppo salp ::: $(seq 0 N)`



