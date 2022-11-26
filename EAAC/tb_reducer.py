from matplotlib import pyplot as plt
import numpy as np
import tensorboard_reducer as tbr
from scipy.interpolate import interp1d

mean = {}
std = {}

# Walker-run
# SAC runs with 200 gradient steps per 100 env steps
# "SAC": ["/data/scratch/idanshen/EAAC/outputs/walker/2022-11-17_21-52-59/SAC_1",
#         "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-17_20-39-07/SAC_1",
#         "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-17_17-32-42/SAC_1",
#         "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-18_09-56-11/SAC_1",
#         "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-18_09-58-53/SAC_1",
#         ],
input_event_dirs = {
                    "EIPO_SAC": ["/home/idanshen/EAAC/outputs/walker/2022-11-22_14-02-54/run_1",
                                 "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-22_15-02-01/run_1",
                                 "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-22_15-04-47/run_1",
                                 "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-22_15-05-51/run_1",
                                 "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-22_15-06-27/run_1"
                    ],
                    "data collection by best policy": ["/home/idanshen/EAAC/outputs/walker/2022-11-25_14-44-52/run_1",],
                    "SAC": [
                        "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-25_18-19-40/SAC_1",
                        "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-25_18-19-57/SAC_1",
                        "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-25_18-20-06/SAC_1",
                        "/data/scratch/idanshen/EAAC/outputs/walker/2022-11-25_18-20-12/SAC_1"
                    ],
                    }
# Reacher-hard

# SAC runs with 200 gradient steps per 100 env steps
# "SAC": ["/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-18_10-08-13/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-18_10-10-10/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-18_12-55-45/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-18_12-56-48/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-18_12-58-37/SAC_1",
#                     ],

# input_event_dirs = {
#                     "EIPO_SAC": ["/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-23_11-30-58/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-23_11-31-09/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-23_11-31-14/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-23_11-31-18/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-24_16-43-42/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-24_16-44-06/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-24_16-44-10/run_1",
#                                  "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-24_16-44-17/run_1",
#                                  ],
#                     "SAC": ["/home/idanshen/EAAC/outputs/reacher/2022-11-24_16-43-15/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-25_11-50-51/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-25_11-50-47/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-25_11-50-38/SAC_1",
#                             "/data/scratch/idanshen/EAAC/outputs/reacher/2022-11-25_11-50-31/SAC_1",
#                             ]
#                     }
x = np.linspace(0, 900000, 201)
stats = {}
for key in input_event_dirs.keys():
    stats[key] = []
    for exp in input_event_dirs[key]:
        events_dict = tbr.load_tb_events([exp])
        data = events_dict["rollout/ep_rew_mean"]
        # data = events_dict["train/ent_coef"]
        index = np.concatenate([[0], np.array(data.index)], axis=0)
        values = np.concatenate([data.values[0], np.array(data.values).squeeze()], axis=0)
        fit = interp1d(index, values)
        stats[key].append(fit(x))
    stats[key] = np.array(stats[key])
    mean[str(key)] = stats[key].mean(axis=0)
    std[str(key)] = stats[key].std(axis=0)

plt.clf()
colors = ['#3F7F4C', '#1B2ACC', '#CC4F1B']
facecolors = ['#7EFF99', '#089FFF', '#FF9848']
for i, k in enumerate(mean.keys()):
    y = np.array(mean[k])
    error = np.array(std[k])
    plt.plot(x, y, 'k', color=colors[i], label=k)
    plt.fill_between(x, y - error, y + error,
                     alpha=0.3, facecolor=facecolors[i],
                     linewidth=0)
    plt.xlabel("env steps")
    plt.ylabel("cumulative reward")
plt.legend()
# plt.axhline(y = 0.0, color = 'b', linestyle = 'dashed')
# plt.yscale("log")
plt.show()
print("end")