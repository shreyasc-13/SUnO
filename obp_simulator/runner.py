import sys
sys.path.insert(1, f'{os.path.abspath(os.getcwd())}/../')

from simulator import OBPSimulator
from utils.metrics import Metrics
from utils.estimators import Estimators
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import yaml
from munch import munchify

###########################################################################
EXP_DIR = '../exps/obp/'

conf = munchify(yaml.safe_load(open('settings.yaml')))

# Simulator args
env = conf.env
num_slots = env.num_slots
num_actions = env.num_actions

alg = conf.alg
sample_size = alg.sample_size
num_runs = alg.num_runs
eval_optimal = alg.eval_optimal # Flag that makes the evaluation policy a little more suboptimal

obp = conf.obp
context_size = obp.context_size
reward_structure = obp.reward_structure
reward_type = obp.reward_type

# Estimator args
est = conf.estimators
num_bins = est.num_bins
normalized = est.normalized

# Experiment args
exp = conf.exp
###########################################################################


def run(num_runs):
    uno_bins = []
    uno_cdfs = []
    suno_cdfs = []
    suno_bins = []
    eval_rs = []

    for i in tqdm(range(num_runs)):

        sim.random_state = i

        eval_r = sim.collect_data()
        sim._define_policies()
        wts = sim._compute_y_k()
        IW, G, rewards = sim.get_weights(wts)
        
        eval_rs.append(eval_r)

        uno_cdf, uno_bin = estimators.UnO(IW, rewards)
        uno_cdfs.append(uno_cdf)
        uno_bins.append(uno_bin)

        suno_cdf, suno_bin = estimators.SUnO(G, rewards)
        suno_cdfs.append(suno_cdf)
        suno_bins.append(suno_bin)

    return (uno_bins, uno_cdfs), (suno_bins, suno_cdfs), eval_rs

def _compute_run_means(uno, suno):
    uno_bins, uno_cdfs = uno
    suno_bins, suno_cdfs = suno

    uno_means = []
    suno_means = []
    for i in range(num_runs):
        uno_means.append(metrics.mean(uno_cdfs[i], uno_bins[i]))
        suno_means.append(metrics.mean(suno_cdfs[i], suno_bins[i]))
    
    return uno_means, suno_means


def print_results(uno_means, suno_means, eval_rs):
    print(f'UnO mean estimate: {np.mean(uno_means):.4f}; Estimate variance: {np.var(uno_means):.4e}')
    print(f'UnO mean estimate: {np.mean(suno_means):.4f}; Estimate variance: {np.var(suno_means):.4e}')
    print(f'True evaluation policy mean reward: {np.mean(eval_rs):.4f}; Variance: {np.var(eval_rs):.4e}')

    s = f'UnO mean estimate: {np.mean(uno_means):.4f}; Estimate variance: {np.var(uno_means):.4e} \n \
        SUnO mean estimate: {np.mean(suno_means):.4f}; Estimate variance: {np.var(suno_means):.4e} \n \
        True evaluation policy mean reward: {np.mean(eval_rs):.4f}; Variance: {np.var(eval_rs):.4e}'

    return s

def log_data(uno, suno, eval_rs):
    with open(f'{EXP_DIR}{exp.name}/uno_cdfs.pkl', 'wb') as f:
        pickle.dump(uno, f)
    with open(f'{EXP_DIR}{exp.name}/suno_cdfs.pkl', 'wb') as f:
        pickle.dump(suno, f)
    with open(f'{EXP_DIR}{exp.name}/eval_rewards.pkl', 'wb') as f:
        pickle.dump(eval_rs, f)
    print(f"Logged data for experiment: {exp.name}")


def plot_data(uno, suno):
    _ = plt.figure()
    # import pdb; pdb.set_trace()
    for b, c in zip(uno[0], uno[1]):
        plt.plot(b[1:], c)
        plt.title('UnO CDFs -- multiple runs')
        plt.xlabel('Rewards')
    plt.savefig(f'{EXP_DIR}{exp.name}/plots/uno_cdfs.png')
    
    _ = plt.figure()
    for b, c in zip(suno[0], suno[1]):
        plt.plot(b[1:], c)
        plt.title('SUnO CDFs -- multiple runs')
        plt.xlabel('Rewards')
    plt.savefig(f'{EXP_DIR}{exp.name}/plots/suno_cdfs.png')

    _ = plt.figure()
    uno_mean = np.mean(np.array(uno[1]), axis=0)
    uno_std = np.sqrt(np.var(np.array(uno[1]), axis=0))
    plt.plot(uno[0][0][1:], uno_mean)
    plt.fill_between(uno[0][0][1:], uno_mean + uno_std, uno_mean - uno_std, alpha = 0.3)
    plt.title('Avg CDF -- UnO')
    plt.xlabel('Rewards')
    plt.savefig(f'{EXP_DIR}{exp.name}/plots/uno_avg_cdf.png')

    _ = plt.figure()
    suno_mean = np.mean(np.array(suno[1]), axis=0)
    suno_std = np.sqrt(np.var(np.array(suno[1]), axis=0))
    plt.plot(suno[0][0][1:], suno_mean)
    plt.fill_between(suno[0][0][1:], suno_mean + suno_std, suno_mean - suno_std, alpha = 0.3)
    plt.title('Avg CDF -- SUnO')
    plt.xlabel('Rewards')
    plt.savefig(f'{EXP_DIR}{exp.name}/plots/suno_avg_cdf.png')
    
    
###########################################################################

# Define the simulator object
sim = OBPSimulator(num_slots, num_actions, sample_size, context_size, reward_structure, reward_type, eval_optimal)

# Estimators object
estimators = Estimators(num_bins, normalized)

# Metrics object
metrics = Metrics()
###########################################################################

# Run and log
uno, suno, eval_rs = run(num_runs)
if exp.log_data or exp.plot_data:
    if os.path.exists(f'{EXP_DIR}{exp.name}'):
        print('Overwriting logs')
    else:
        os.makedirs(f'{EXP_DIR}{exp.name}')
        os.makedirs(f'{EXP_DIR}{exp.name}/plots')
        
uno_means, suno_means = _compute_run_means(uno, suno)
print_results(uno_means, suno_means, eval_rs)

if exp.log_data:
    log_data(uno, suno, eval_rs)
    with open(f'{EXP_DIR}{exp.name}/setting.json', 'w') as f:
        f.write(conf.toJSON())
        f.write('\n')
        f.write(print_results(uno_means, suno_means, eval_rs))

if exp.plot_data:
    plot_data(uno, suno)