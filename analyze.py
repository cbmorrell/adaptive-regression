import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
from libemg.environments.controllers import RegressorController


def in_circle(cursor, circle):
    return math.sqrt((circle[0] - cursor[0])**2 + (circle[1] - cursor[1])**2) < circle[2]/2 + cursor[2]/2


def is_timeout_trial(trial, run_log):
    timeout_trials = find_timeout_trials(run_log)
    return trial in timeout_trials

def calculate_efficiency(run_log):
    efficiency = []
    trials = np.unique(run_log['trial_number'])
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        distance_travelled = np.sum([math.dist(run_log['cursor_position'][t_idxs[i]][0:2], run_log['cursor_position'][t_idxs[i-1]][0:2]) for i in range(1,len(t_idxs))])
        fastest_path = math.dist((run_log['cursor_position'][t_idxs[0]])[0:2], (run_log['goal_circle'][t_idxs[0]])[0:2])
        efficiency.append(fastest_path/distance_travelled)
    return np.mean(efficiency)


def calculate_throughput(run_log):
    # See https://www.yorku.ca/mack/hhci2018.html for explanation of effective width and TP calculation
    raise NotImplementedError('Need to make sure this factors in the width of the target.')
    trials = np.unique(run_log['trial_number'])
    effective_amplitudes = []
    effective_distances = []
    trial_times = []
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        trial_time = run_log['global_clock'][t_idxs[-1]] - run_log['global_clock'][t_idxs[0]]
        trial_times.append(trial_time)
        starting_cursor_position = (run_log['cursor_position'][t_idxs[0]])[0:2]
        goal_circle_position = (run_log['goal_circle'][t_idxs[0]])[0:2]
        final_cursor_position = (run_log['cursor_position'][t_idxs[-1]])[0:2]  # maybe take the average across the dwell time if we have time
        a = math.dist(starting_cursor_position, goal_circle_position)
        b = math.dist(final_cursor_position, goal_circle_position)
        c = math.dist(starting_cursor_position, final_cursor_position)
        dx = (c * c - b * b - a * a) / (2 * a)
        ae = a + dx
        effective_amplitudes.append(ae)
        effective_distances.append(dx)

    if len(effective_amplitudes) < 2:
        # Return throughput of 0 if acquired less than 2 targets
        return 0
    run_effective_ampitude = np.mean(effective_amplitudes)
    run_effective_width = np.std(effective_distances) * 4.333
    effective_id = math.log2((run_effective_ampitude / run_effective_width) + 1)
    movement_time = np.mean(trial_times)

    return effective_id / movement_time


def calculate_overshoots(run_log):
    overshoots = []
    trials = np.unique(run_log['trial_number'])
    for t in trials:
        if is_timeout_trial(t, run_log):
            # Ignore trial
            continue
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        cursor_locs = np.array(run_log['cursor_position'])[t_idxs]
        targets = np.array(run_log['goal_circle'])[t_idxs]
        in_bounds = [in_circle(cursor_locs[i], targets[i]) for i in range(0,len(cursor_locs))]
        trial_overshoots = 0
        for i in range(1,len(in_bounds)):
            if in_bounds[i-1] == True and in_bounds[i] == False:
                trial_overshoots += 1 
        overshoots.append(trial_overshoots)
    return sum(overshoots)

def find_timeout_trials(run_log):
    trials = np.unique(run_log['trial_number'])
    timeout_trials = []
    for t in trials:
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        trial_time = run_log['global_clock'][t_idxs[-1]] - run_log['global_clock'][t_idxs[0]]
        if trial_time > 30:
            # Went to timeout
            timeout_trials.append(t)
    return timeout_trials


def extract_traces(run_log):
    trials = np.unique(run_log['trial_number'])
    traces = []
    for t in trials:
        t_idxs = np.where(run_log['trial_number'] == t)[0]
        traces.append(np.array(run_log['cursor_position'])[t_idxs][:, :2])
    return traces


def extract_model_predictions(run_log):
    controller = RegressorController()
    trials = np.unique(run_log['trial_number'])
    predictions = []
    for t in trials:
        trial_mask = np.where(run_log['trial_number'] == t)[0]
        model_outputs = np.array(run_log['class_label'])[trial_mask]
        model_outputs = [controller._parse_predictions(model_output) for model_output in model_outputs]
        predictions.append(model_outputs)
    return predictions


def extract_fitts_metrics(run_log):
    fitts_results = {}
    fitts_results['timeouts'] = find_timeout_trials(run_log)
    fitts_results['overshoots'] = calculate_overshoots(run_log)
    fitts_results['efficiency'] = calculate_efficiency(run_log)
    fitts_results['throughput'] = calculate_throughput(run_log)
    return fitts_results


def main():
    with open('/Users/cmorrell/Code/adaptive-regression/data/subject-003/ciil/val_fitts.pkl', 'rb') as f:
        ciil_data = pickle.load(f)

    predictions = np.linalg.norm(ciil_data['current_direction'], axis=1) / 25
    target_positions = np.array(ciil_data['goal_circle'])[:, :2]
    cursor_positions = np.array(ciil_data['cursor_position'])[:, :2]
    distances = np.linalg.norm(target_positions - cursor_positions, axis=1)

    plt.figure()
    plt.scatter(distances, predictions)
    
    ciil_results = extract_fitts_metrics(ciil_data)


    with open('/Users/cmorrell/Code/adaptive-regression/data/subject-003/oracle/val_fitts.pkl', 'rb') as f:
        oracle_prop_data = pickle.load(f)
    with open('/Users/cmorrell/Code/adaptive-regression/data/subject-003/oracle-no-prop/val_fitts.pkl', 'rb') as f:
        oracle_no_prop_data = pickle.load(f)

    oracle_prop_results = extract_fitts_metrics(oracle_prop_data)
    oracle_no_prop_results = extract_fitts_metrics(oracle_no_prop_data)


    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 6), layout='constrained')
    x = ['ciil', 'oracle (distance)', 'oracle (prediction)']
    axs[0].bar(x, [ciil_results['throughput'], oracle_prop_results['throughput'], oracle_no_prop_results['throughput']])
    axs[1].bar(x, [ciil_results['efficiency'], oracle_prop_results['efficiency'], oracle_no_prop_results['efficiency']])
    axs[2].bar(x, [ciil_results['overshoots'], oracle_prop_results['overshoots'], oracle_no_prop_results['overshoots']])

    axs[0].set_ylabel('Throughput')
    axs[1].set_ylabel('Path Efficiency')
    axs[2].set_ylabel('Overshoots')

    plt.show()
    print('-------------Analyze complete!-------------')


if __name__ == '__main__':
    main()
