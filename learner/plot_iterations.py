import matplotlib.pyplot as plt
import pickle as pk
import json
import os

# from .utils.helper import FDMeta

folder_path = "./store"
folders = os.listdir(folder_path)

figure = plt.figure()
for folder in folders:
    with open(os.path.join(folder_path, folder, "project_info.json"), 'r') as fp:
        project_info = json.load(fp)
        target_fd = project_info["scenario"]["target_fd"]
    path = os.path.join(folder_path, folder, "fd_metadata.p")
    if os.path.exists(path):
        with open(path, 'rb') as fp:
            data = pk.load(fp)
            print(data)
        
        for fd, fd_metadata in data.items():
            plot_data= [(d.iter_num,d.value) for d in fd_metadata.conf_history]
            print(plot_data)
            if fd == target_fd:
                plt.plot([x for (x,y) in plot_data], [y for (x,y) in plot_data], 'r--')
            else:
                plt.plot([x for (x,y) in plot_data], [y for (x,y) in plot_data], 'b--')

plt.xlabel("Iterations")
plt.ylabel('FD confidence')
plt.title("Random Sampling with Resampling for scenario 1")
plt.show()
