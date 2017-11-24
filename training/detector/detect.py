import multiprocessing as mp
import time

import numpy as np

from layers import acc

save_dir = 'results/ma_offset40_res_n6_100-1/'
pbb = np.load(save_dir + 'pbb.npy')
lbb = np.load(save_dir + 'lbb.npy')

conf_th = [-1, 0, 1]
nms_th = [0.3, 0.5, 0.7]
detect_th = [0.2, 0.3]


def mp_get_pr(conf_th, nms_th, detect_th, num_procs=64):
    start_time = time.time()

    num_samples = len(pbb)
    split_size = int(np.ceil(float(num_samples) / num_procs))
    num_procs = int(np.ceil(float(num_samples) / split_size))

    manager = mp.Manager()
    tp = manager.list(range(num_procs))
    fp = manager.list(range(num_procs))
    p = manager.list(range(num_procs))
    procs = []
    for pid in range(num_procs):
        proc = mp.Process(
            target=get_pr,
            args=(
                pbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                lbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                conf_th, nms_th, detect_th, pid, tp, fp, p))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    tp = np.sum(tp)
    fp = np.sum(fp)
    p = np.sum(p)

    end_time = time.time()
    print('conf_th %1.1f, nms_th %1.1f, detect_th %1.1f, tp %d, fp %d, p %d, recall %f, time %3.2f' % (
    conf_th, nms_th, detect_th, tp, fp, p, float(tp) / p, end_time - start_time))


def get_pr(pbb, lbb, conf_th, nms_th, detect_th, pid, tp_list, fp_list, p_list):
    tp, fp, p = 0, 0, 0
    for i in range(len(pbb)):
        tpi, fpi, pi = acc(pbb[i], lbb[i], conf_th, nms_th, detect_th)
        tp += tpi
        fp += fpi
        p += pi
    tp_list[pid] = tp
    fp_list[pid] = fp
    p_list[pid] = p


if __name__ == '__main__':
    for ct in conf_th:
        for nt in nms_th:
            for dt in detect_th:
                mp_get_pr(ct, nt, dt)
