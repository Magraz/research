### To Install
Project requires Python 3.12

Run the following commands inside after cloning:
1. `git submodule init`
2. `git submodule update`
3. `cd VectorizedMultiAgentSimulator`
4. `pip install -e .`
7. `cd ../src`
8. `pip install -e .`
9. `cd ..`
10. `pip install -r requirements.txt`

### To Run Experiment

Run the following command for running experiments in different modalities:

- `python3 src/run_trial.py --batch=salp_navigate_8a --name=gcn --algorithm=ppo --environment=salp_navigate`

Run N number of trials in parallel (Requires GNU Parallel Package)

- `parallel bash run_trial.sh salp_navigate_8a ppo salp_navigate test_id ::: gcn gat graph_transformer transformer_full transformer_encoder transformer_decoder`



