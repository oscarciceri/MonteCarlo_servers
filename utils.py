import json
import numpy as np
import scipy.stats
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def argmax(Q, s):
    # pares estado-ação que tem estado = s
    pares = list(filter(lambda step: step[0] == s, list(Q.keys())))
    sub_Q = {k: v for k, v in Q.items() if k in pares}

    top_value = float("-inf")
    ties = []

    for i in sub_Q.items():
        # if a value in q_values is greater than the highest value update top
        if i[1] > top_value:
            top_value = i[1]

    # if a value is equal to top value add to ties
    ties = list(filter(lambda step: step[1] == top_value, list(sub_Q.items())))

    if len(ties) > 0:
        # chose the random index of the ties
        chosen = np.random.choice(range(len(ties)))
    else:
        return 0, False

    return ties[chosen][0][1], True


# def argmax(q_values, s):

#     top_value = float("-inf")
#     ties = []

#     for i in range(len(q_values[s])):
#         # if a value in q_values is greater than the highest value update top
#         if q_values[s, i] > top_value:
#             top_value = q_values[s, i]

#     # if a value is equal to top value add the index to ties
#     ties = np.where(np.array(q_values[s, :]) == top_value)[0]

#     if len(ties) > 0:
#         # return a random selection from ties.
#         return np.random.choice(ties), True
#     else:
#         return 0, False


def write_json(content, file_name):
    with open(file_name + '.json', 'w') as json_file:
        json.dump(content, json_file, separators=(',', ':'), ensure_ascii=False, sort_keys=True, indent=4)


def open_file(name_file):
    try:
        f = open('../' + name_file, "r")
        data = json.loads(f.read())
        f.close()
    except:
        data = 0
        pass

    return data


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def average_cost(files_result, directory, confidence=0.95):
    all = []
    err = []

    for a in files_result:
        costs = []
        data = dict(open_file(directory + a))
        # print(data)
        for i in list(data.keys()):
            costs.append(float(data.get(str(i))))
        m, h = mean_confidence_interval(costs, confidence)
        all.append(m)
        err.append(h)
    return all, err


def plot_all_paths_policy(G, policy, source, target):
    """_summary_

    :param G: network
    :type G: networkx.Graph
    :param policy: dict with nodes in keys and the action in values
    :type policy: dict
    :param source: source node
    :type source: int
    :param target: target node
    :type target: int
    """

    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
        alpha=0.25
    )

    for i in G.nodes:
        dest = policy[i]

        # verify if there is a edge between i and dest, if not, continue
        if (i, dest) not in G.edges:
            continue

        plt.arrow(
            G.nodes[i]["x"],
            G.nodes[i]["y"],
            (G.nodes[dest]["x"] - G.nodes[i]["x"]) * 0.7,
            (G.nodes[dest]["y"] - G.nodes[i]["y"]) * 0.7,
            alpha=0.8,
            width=0.0001
        )

    # plot the source and target
    plt.scatter(
        [G.nodes[source]["x"], G.nodes[target]["x"]],
        [G.nodes[source]["y"], G.nodes[target]["y"]],
        color=["green", "red"],
        alpha=1,
        s=100
    )

    # add legend to source and target with the color

    red_patch = mpatches.Patch(color='red', label='Target')
    green_patch = mpatches.Patch(color='green', label='Source')
    plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_few_steps_policy(G, policy, source, target, steps=5):
    """_summary_

    :param G: network
    :type G: networkx.Graph
    :param policy: dict with nodes in keys and the action in values
    :type policy: dict
    :param source: source node
    :type source: int
    :param target: target node
    :type target: int
    :param steps: number of steps to plot, defaults to 5
    :type steps: int, optional
    """

    plt.scatter(
        [G.nodes[i]["x"] for i in G.nodes],
        [G.nodes[i]["y"] for i in G.nodes],
        alpha=0.25
    )

    # plot the edges
    for u, v, m in G.edges:
        if m == 0:
            plt.plot(
                [G.nodes[u]["x"], G.nodes[v]["x"]],
                [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="black",
                alpha=0.25
            )

    state = source
    for i in range(steps):
        dest = policy[state]

        # verify if there is a edge between i and dest, if not, continue
        if (state, dest) not in G.edges:
            raise ValueError(f"There is no edge between {state} and {dest}")

        plt.arrow(
            G.nodes[state]["x"],
            G.nodes[state]["y"],
            (G.nodes[dest]["x"] - G.nodes[state]["x"]) * 0.95,
            (G.nodes[dest]["y"] - G.nodes[state]["y"]) * 0.95,
            alpha=0.8,
            width=0.0001
        )
        state = dest

    # plot the source and target
    plt.scatter(
        [G.nodes[source]["x"], G.nodes[target]["x"]],
        [G.nodes[source]["y"], G.nodes[target]["y"]],
        color=["green", "red"],
        alpha=1,
        s=100
    )

    # add legend to source and target with the color

    red_patch = mpatches.Patch(color='red', label='Target')
    green_patch = mpatches.Patch(color='green', label='Source')
    plt.legend(handles=[red_patch, green_patch])

    plt.xticks([])
    plt.yticks([])


def analysis_results(reward_method, env_method):
    results_lr = pd.read_json(f"results/{reward_method}_{env_method}_lr.json")
    results_gamma = pd.read_json(f"results/{reward_method}_{env_method}_gamma.json")
    results_lr["rewards"] = results_lr.rewards.apply(np.mean)
    results_gamma["rewards"] = results_gamma.rewards.apply(np.mean)

    results_lr = results_lr.groupby("learning_rate").agg({
        "cost": ["mean", "std"],
        "time": ["mean", "std"],
        "rewards": ["mean", "std"]
    })
    results_lr.columns = ["_".join(col) for col in results_lr.columns]
    results_lr = results_lr.reset_index()

    results_gamma = results_gamma.groupby("gamma").agg({
        "cost": ["mean", "std"],
        "time": ["mean", "std"],
        "rewards": ["mean", "std"]
    })
    results_gamma.columns = ["_".join(col) for col in results_gamma.columns]
    results_gamma = results_gamma.reset_index()
    plot_results_experiments(results_lr, results_gamma, f"{reward_method} reward and {env_method} environment")


def plot_results_experiments(results_lr, results_gamma, title):
    """Plot the results of the experiments varying learning rate and gamma.

    Parameters
    ----------
    results_lr : pd.DataFrame
        Dataframe with columns learning_rate, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    results_gamma : pd.DataFrame
        Dataframe with columns gamma, cost_mean, cost_std, time_mean, time_std, rewards_mean, rewards_std
    title : str
        Title of the plot
    """

    def plot_line_plot_err(df, param, cost, ax):
        ax.plot(df[param], df[f"{cost}_mean"], color="black")
        ax.fill_between(
            df[param],
            df[f"{cost}_mean"] - df[f"{cost}_std"],
            df[f"{cost}_mean"] + df[f"{cost}_std"],
            color="black",
            alpha=0.2
        )
        ax.set_xlabel(param)
        ax.set_ylabel(cost)
        ax.set_title(f"{cost} by {param}")
        ax.grid()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    axs = axs.flatten()

    plot_line_plot_err(results_lr, "learning_rate", "cost", axs[0])
    plot_line_plot_err(results_lr, "learning_rate", "time", axs[1])
    plot_line_plot_err(results_lr, "learning_rate", "rewards", axs[2])

    plot_line_plot_err(results_gamma, "gamma", "cost", axs[3])
    plot_line_plot_err(results_gamma, "gamma", "time", axs[4])
    plot_line_plot_err(results_gamma, "gamma", "rewards", axs[5])

    plt.suptitle(title, fontsize=16)
    plt.show()