"""B10 phan phu - cung chi muc ANN do choi, tren du lieu tong hop o BA muc chong lan cum.
Muc dich: cho thay DO PHU do tren du lieu tong hop phu thuoc hoan toan vao cach sinh du lieu."""
import json, math, statistics, time, os
import numpy as np
rng = np.random.default_rng(0)
def chuan(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
def topk(s, k): i = np.argpartition(-s, k)[:k]; return i[np.argsort(-s[i])]
class IVF:
    def __init__(self, nlist, iters=8): self.nlist, self.iters = nlist, iters
    def dung(self, X):
        m = X[rng.choice(len(X), 20000, replace=False)]; c = m[rng.choice(len(m), self.nlist, replace=False)].copy()
        for _ in range(self.iters):
            a = (m @ c.T).argmax(1)
            for j in range(self.nlist):
                s = m[a == j]
                if len(s): c[j] = s.mean(0)
            c = chuan(c)
        self.c = c; a = (X @ c.T).argmax(1)
        self.ds = [np.where(a == j)[0] for j in range(self.nlist)]; self.Xl = [X[d] for d in self.ds]; return self
    def tim(self, q, k, npb):
        l = topk(self.c @ q, npb); ids = np.concatenate([self.ds[j] for j in l])
        s = np.concatenate([self.Xl[j] @ q for j in l]); return ids[topk(s, k)] if len(ids) > k else ids, len(ids)
N, D, C = 100_000, 384, 1000
res = {}
for nhieu in (0.6, 0.9, 1.2):
    tam = chuan(rng.standard_normal((C, D))); sig = nhieu / math.sqrt(D)
    X = chuan(tam[rng.integers(0, C, N)] + sig * rng.standard_normal((N, D))).astype(np.float32)
    Q = chuan(tam[rng.integers(0, C, 300)] + sig * rng.standard_normal((300, D))).astype(np.float32)
    ex = [topk(X @ q, 10) for q in Q]
    t = [time.perf_counter()]; [X @ q for q in Q[:100]]; ms_ex = (time.perf_counter() - t[0]) / 100 * 1e3
    ivf = IVF(316).dung(X); r = dict(ms_quet_toan_bo=round(ms_ex, 3), ivf={})
    for npb in (1, 2, 4, 8, 16, 32):
        ts, ph = [], []
        for q, e in zip(Q, ex):
            t0 = time.perf_counter(); a, _ = ivf.tim(q, 10, npb); ts.append(time.perf_counter() - t0)
            ph.append(len(set(a) & set(e)) / 10)
        r["ivf"][npb] = dict(ms_p50=round(statistics.median(ts) * 1e3, 3), do_phu_at10=round(statistics.mean(ph), 3))
    res[nhieu] = r; print("nhieu", nhieu, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b10b.json"), "w"), indent=1)
