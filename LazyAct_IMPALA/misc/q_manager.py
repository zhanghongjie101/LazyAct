import torch
import time
import numpy as np
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QManager(object):
    """
    single-machine implementation
    """

    def __init__(self, args, q_trace, q_batch):
        self.q_trace = q_trace
        self.q_batch = q_batch
        self.args = args

    def listening(self):
        res_s, res_a, res_k, res_r, res_c, res_p, res_kp, res_masks, skip_ratio = [],[],[],[],[],[],[],[],[]
        while True:
            trace = self.q_trace.get(block=True)
            # in
            res_s.append(trace[0])
            res_a.append(trace[1])
            res_k.append(trace[2])
            res_r.append(trace[3])
            res_c.append(trace[4])
            res_p.append(trace[5])
            res_kp.append(trace[6])
            res_masks.append(trace[7])
            skip_ratio.append(trace[8])

            # produce_batch
            if len(res_s) >= self.args.batch_size:
                torch.cuda.empty_cache()
                self.q_batch.put((torch.from_numpy(np.stack(res_s, axis=0).astype(np.float32)).to(device), \
                    torch.from_numpy(np.stack(res_a, axis=0)).to(device), \
                    torch.from_numpy(np.stack(res_k, axis=0)).to(device), \
                    torch.from_numpy(np.stack(res_r, axis=0).astype(np.float32)).to(device), \
                    torch.from_numpy(np.stack(res_c, axis=0).astype(np.float32)).to(device), \
                    torch.from_numpy(np.stack(res_p, axis=0).astype(np.float32)).to(device), \
                    torch.from_numpy(np.stack(res_kp, axis=0).astype(np.float32)).to(device), \
                    torch.from_numpy(np.stack(res_masks, axis=0).astype(np.float32)).to(device), \
                    torch.FloatTensor([np.mean(skip_ratio).astype(np.float32)]).to(device)))
                res_s, res_a, res_k, res_r, res_c, res_p, res_kp, res_masks, skip_ratio = [],[],[],[],[],[],[],[],[]
                torch.cuda.empty_cache()
