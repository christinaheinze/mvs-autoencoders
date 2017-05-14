import os
import re
import numpy as np

def get_dirs2(save_dir, log_dir, filterBy):
    all_log_dirs = listdir_fullpath(log_dir)
    logdir = [d for d in all_log_dirs if filterBy in d]
    if(len(logdir) > 1):
        print("more than one logdir found")
    else:
        logdir = logdir[0]
    all_folders_dirs = listdir_fullpath(logdir)
    time_stamps = os.listdir(logdir)
    logdir_out = all_folders_dirs[np.argmax(time_stamps)]

    all_save_dirs = listdir_fullpath(save_dir)
    savedir = [d for d in all_save_dirs if filterBy in d]
    if(len(savedir) > 1):
        print("more than one savedir found")
    else:
        savedir = savedir[0]
    return (savedir, logdir_out)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
