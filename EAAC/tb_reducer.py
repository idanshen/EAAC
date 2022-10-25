from matplotlib import pyplot as plt
import numpy as np
import tensorboard_reducer as tbr

mean = {}
std = {}
for i in range(10):
    input_event_dirs = ["/home/idanshen/EAAC/multirun/2022-10-21/18-28-20/"+str(i)+"/SAC_1/",
                        "/home/idanshen/EAAC/multirun/2022-10-22/12-07-22/"+str(i)+"/SAC_1/",
                        "/home/idanshen/EAAC/multirun/2022-10-23/01-37-08/"+str(i)+"/SAC_1/",
                        "/home/idanshen/EAAC/multirun/2022-10-23/14-29-24/"+str(i)+"/SAC_1/",
                        "/home/idanshen/EAAC/multirun/2022-10-24/14-47-03/"+str(i)+"/SAC_1/"
                        ]
    # where to write reduced TB events, each reduce operation will be in a separate subdirectory
    tb_events_output_dir = "/home/idanshen/EAAC/multirun/"
    csv_out_path = "/home/idanshen/EAAC/multirun/reduced_"+str(i)+".csv"
    # whether to abort or overwrite when csv_out_path already exists
    overwrite = False
    reduce_ops = ("mean", "std")

    events_dict = tbr.load_tb_events(input_event_dirs)

    # number of recorded tags. e.g. would be 3 if you recorded loss, MAE and R^2
    n_scalars = len(events_dict)
    n_steps, n_events = list(events_dict.values())[0].shape

    print(
        f"Loaded {n_events} TensorBoard runs with {n_scalars} scalars and {n_steps} steps each"
    )
    print(", ".join(events_dict))

    reduced_events = tbr.reduce_events(events_dict, reduce_ops)
    mean[str(i-10)] = reduced_events["mean"]["rollout/ep_rew_mean"]
    std[str(i-10)] = reduced_events["std"]["rollout/ep_rew_mean"]
    # for op in reduce_ops:
    #     print(f"Writing '{op}' reduction to '{tb_events_output_dir}-{op}'")

    # tbr.write_tb_events(reduced_events, tb_events_output_dir, overwrite)

    # print(f"Writing results to '{csv_out_path}'")
    #
    # tbr.write_data_file(reduced_events, csv_out_path, overwrite)
    #
    # print("Reduction complete")
plt.clf()
for j, color, facecolor in zip(['-2', '-6', '-10'], ['#3F7F4C','#1B2ACC','#CC4F1B'], ['#7EFF99', '#089FFF', '#FF9848']):
    x = np.array(mean[j].index)
    y = np.array(mean[j].values)
    error = np.array(std[j].values)
    plt.plot(x, y, 'k', color =color,label=j)
    plt.fill_between(x, y-error, y+error,
        alpha=0.3, facecolor=facecolor,
        linewidth=0)
plt.legend()
plt.show()
print("end")