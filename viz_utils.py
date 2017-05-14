import os
import re
import numpy as np

def get_dirs(res_dir):
    all_dirs_w_log = os.listdir(res_dir)
    all_dirs = [d for d in all_dirs_w_log if not re.search("log", d)]
    save_dirs_all = []
    ckpt_dirs_all = []
    for top_level_dir_no in range(len(all_dirs)):
        log_dir = all_dirs[top_level_dir_no]+"_log"
        path_save_dir = os.path.join(res_dir, all_dirs[top_level_dir_no])
        path_ckpt_top = os.path.join(res_dir, log_dir)
        f = []
        g = []
        for (dirpath, dirnames, filenames) in os.walk(path_save_dir):
            dirnames_full = [os.path.join(path_save_dir, d) for d in dirnames]
            dirnames_full_ckpt = [os.path.join(path_ckpt_top, d) for d in dirnames]
            f.extend(dirnames_full)
            g.extend(dirnames_full_ckpt)

        subdirs = [os.listdir(direc) for direc in g]

        ckpt_dirs = []
        for direc in range(len(g)):
            ckpt_dirs.extend([os.path.join(g[direc], s) for s in subdirs[direc]])

        save_dirs_all.extend(f)
        ckpt_dirs_all.extend(ckpt_dirs)

    return(save_dirs_all, ckpt_dirs_all)

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
