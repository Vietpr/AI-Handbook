"""B10 - Kho vector khi van hanh.
(1) chi muc ANN do choi tren 100.000 vector TONG HOP co cau truc cum -> do tre he thong, KHONG phai chat luong ngu nghia
(2) cung chi muc tren kho that (toan bo loat bai) -> do phu so voi tim chinh xac + article hit@5 theo nhan
(3) van hanh: them chuong moi vao chi muc cu khong huan luyen lai; xoa bang dau mo
(4) doi mo hinh embedding cung 384 chieu -> khong co loi nao, chi co ket qua rac"""
import glob, json, os, time, math, statistics
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
GOC = "/home/v002744/Documents/Book"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b10.json")
torch.set_num_threads(4); rng = np.random.default_rng(0)
res = dict(luc=time.strftime("%Y-%m-%d %H:%M:%S"), threads=4)
def chuan(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
def topk(s, k): i = np.argpartition(-s, k)[:k]; return i[np.argsort(-s[i])]

class IVF:
    """Chi muc phan cum toi gian: k-means cau tren mau, moi cum mot danh sach."""
    def __init__(self, nlist, iters=8): self.nlist, self.iters = nlist, iters
    def huan_luyen(self, X):
        m = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
        c = m[rng.choice(len(m), self.nlist, replace=False)].copy()
        for _ in range(self.iters):
            a = (m @ c.T).argmax(1)
            for j in range(self.nlist):
                s = m[a == j]
                if len(s): c[j] = s.mean(0)
            c = chuan(c)
        self.c = c; return self
    def them(self, X, id0=0):
        a = (X @ self.c.T).argmax(1)
        if not hasattr(self, "ds"): self.ds = [np.empty(0, int) for _ in range(self.nlist)]
        for j in range(self.nlist):
            self.ds[j] = np.concatenate([self.ds[j], np.where(a == j)[0] + id0])
        return self
    def dong_goi(self, Xall):
        self.Xl = [Xall[d] for d in self.ds]; return self
    def tim(self, q, k, nprobe):
        l = topk(self.c @ q, nprobe)
        ids = np.concatenate([self.ds[j] for j in l]); s = np.concatenate([self.Xl[j] @ q for j in l])
        if len(ids) <= k: return ids[np.argsort(-s)], len(ids)
        return ids[topk(s, k)], len(ids)

# ================= (1) TONG HOP =================
N, D, C = 100_000, 384, 1000
tam = chuan(rng.standard_normal((C, D))); sig = 0.6 / math.sqrt(D)
X = chuan(tam[rng.integers(0, C, N)] + sig * rng.standard_normal((N, D))).astype(np.float32)
Q = chuan(tam[rng.integers(0, C, 300)] + sig * rng.standard_normal((300, D))).astype(np.float32)
def do(f):
    ts, out = [], []
    for q in Q:
        t = time.perf_counter(); out.append(f(q)); ts.append(time.perf_counter() - t)
    return out, round(statistics.median(ts) * 1e3, 3), round(sorted(ts)[int(len(ts) * .95)] * 1e3, 3)
chinh_xac, ms50, ms95 = do(lambda q: topk(X @ q, 10))
tong = dict(N=N, D=D, cum_that=C, quet_toan_bo=dict(ms_p50=ms50, ms_p95=ms95, do_phu=1.0))
t = time.perf_counter(); ivf = IVF(316).huan_luyen(X).them(X).dong_goi(X); tong["giay_dung_chi_muc"] = round(time.perf_counter() - t, 1)
tong["ivf"] = {}
for npb in (1, 2, 4, 8, 16, 32, 64):
    kq, a, b = do(lambda q: ivf.tim(q, 10, npb))
    phu = statistics.mean(len(set(r[0]) & set(e)) / 10 for r, e in zip(kq, chinh_xac))
    quet = statistics.mean(r[1] for r in kq) / N
    tong["ivf"][npb] = dict(ms_p50=a, ms_p95=b, do_phu_at10=round(phu, 3), phan_tram_kho_da_quet=round(100 * quet, 2))
    print("tong hop nprobe", npb, tong["ivf"][npb], flush=True)
print("quet toan bo:", tong["quet_toan_bo"], flush=True)
res["tong_hop"] = tong

# ================= (2) KHO THAT =================
CHUONG = ["Foundation", "Machine Learning", "Deep Learning", "Computer Vision", "Generative AI", "AI Systems"]
doan = []
for ch in CHUONG:
    for f in sorted(glob.glob(os.path.join(GOC, ch, "*.md"))):
        t_ = open(f, encoding="utf-8").read(); i = 0
        while i < len(t_):
            d = t_[i:i + 800]
            if len(d) > 150: doan.append((ch, os.path.basename(f), d))
            i += 600
import importlib.util
sp = importlib.util.spec_from_file_location("rag", os.path.join(GOC, "Generative AI/code/rag_sach.py"))
rag = importlib.util.module_from_spec(sp); sp.loader.exec_module(rag)
HOI = list(rag.CAU_HOI_CO_NHAN) + [
    ("Pha nạp prompt và pha sinh token khác nhau thế nào?", "AI Systems", "01"),
    ("TTFT và TPOT là gì?", "AI Systems", "01"),
    ("Điểm cân bằng FLOP trên byte của GPU là gì?", "AI Systems", "01"),
    ("KV cache tốn bao nhiêu byte cho mỗi token?", "AI Systems", "02"),
    ("Grouped-query attention giảm bộ nhớ đệm như thế nào?", "AI Systems", "02"),
    ("Vì sao logits chiếm nhiều bộ nhớ khi nạp prompt?", "AI Systems", "02"),
    ("Vì sao mô hình 7B không vừa card 12 GB?", "AI Systems", "03"),
    ("memory_allocated và memory_reserved khác nhau thế nào?", "AI Systems", "03")]
MOI = [i for i, h in enumerate(HOI) if h[1] == "AI Systems"]

def bo_ma(ten, tien_to_q, tien_to_p):
    tk = AutoTokenizer.from_pretrained(ten); md = AutoModel.from_pretrained(ten).eval()
    @torch.no_grad()
    def ma(xs, tt):
        out = []
        for k in range(0, len(xs), 32):
            b = tk([tt + x for x in xs[k:k + 32]], padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = md(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1))
        return torch.cat(out).numpy().astype(np.float32)
    t = time.perf_counter(); P = ma([d for _, _, d in doan], tien_to_p); g = time.perf_counter() - t
    return P, ma([h for h, _, _ in HOI], tien_to_q), round(g, 1)

A = "intfloat/multilingual-e5-small"
PA, QA, giayA = bo_ma(A, "query: ", "passage: ")
res["kho_that"] = dict(so_doan=len(doan), so_cau_hoi=len(HOI), mo_hinh=A, giay_ma_hoa_ca_kho=giayA,
                       doan_moi_giay=round(len(doan) / giayA, 1))
print("kho that:", res["kho_that"], flush=True)

def hit(ids, i):
    _, ch, so = HOI[i]
    return any(doan[j][0] == ch and doan[j][1].startswith(so + ".") for j in ids)
def danh_gia(P, Qv, chi=None, k=5):
    chi = range(len(HOI)) if chi is None else chi
    return round(statistics.mean(hit(topk(P @ Qv[i], k), i) for i in chi), 3)
ex = [topk(PA @ QA[i], 5) for i in range(len(HOI))]
res["kho_that"]["quet_toan_bo"] = dict(article_hit5=danh_gia(PA, QA),
                                       ms=round(statistics.median([(lambda t0: (PA @ QA[0], time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(50)]) * 1e3, 3))
nl = max(4, round(math.sqrt(len(doan))))
iv = IVF(nl).huan_luyen(PA).them(PA).dong_goi(PA)
res["kho_that"]["ivf"] = {"nlist": nl}
for npb in (1, 2, 4, 8):
    r = [iv.tim(QA[i], 5, npb)[0] for i in range(len(HOI))]
    res["kho_that"]["ivf"][npb] = dict(do_phu_at5=round(statistics.mean(len(set(a) & set(b)) / 5 for a, b in zip(r, ex)), 3),
                                       article_hit5=round(statistics.mean(hit(a, i) for i, a in enumerate(r)), 3))
print("kho that:", res["kho_that"], flush=True)

# ================= (3) VAN HANH =================
cu = np.array([i for i, d in enumerate(doan) if d[0] != "AI Systems"]); moi = np.array([i for i, d in enumerate(doan) if d[0] == "AI Systems"])
iv_cu = IVF(nl); iv_cu.huan_luyen(PA[cu]); iv_cu.ds = [np.empty(0, int) for _ in range(nl)]
a = (PA @ iv_cu.c.T).argmax(1)
for j in range(nl): iv_cu.ds[j] = np.where(a == j)[0]
iv_cu.dong_goi(PA)
kich = np.array([len(d) for d in iv_cu.ds]); ds_moi = set(a[moi].tolist())
def phu_moi(idx, npb):
    return round(statistics.mean(len(set(idx.tim(QA[i], 5, npb)[0]) & set(ex[i])) / 5 for i in MOI), 3)
vh = dict(so_doan_chuong_moi=int(len(moi)), so_cum_nhan_doan_moi=len(ds_moi),
          danh_sach_lon_nhat_chia_trung_binh=round(kich.max() / kich.mean(), 2))
for npb in (1, 2):
    vh[f"khong_huan_luyen_lai_nprobe{npb}"] = phu_moi(iv_cu, npb)
    vh[f"huan_luyen_lai_nprobe{npb}"] = phu_moi(iv, npb)
xoa = set(rng.choice(len(doan), int(0.3 * len(doan)), replace=False).tolist())
def thieu(kq):
    return sum(len([j for j in iv.tim(QA[i], kq, 2)[0] if j not in xoa][:5]) < 5 for i in range(len(HOI)))
vh["xoa_30pt_dau_mo"] = dict(so_cau_tra_ve_thieu_5_khi_lay_5=thieu(5), khi_lay_du_10=thieu(10), khi_lay_du_20=thieu(20),
                             tong_cau=len(HOI))
res["van_hanh"] = vh; print("van hanh:", vh, flush=True)

# ================= (4) DOI MO HINH EMBEDDING =================
B = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PB, QB, giayB = bo_ma(B, "", "")
res["doi_mo_hinh"] = dict(mo_hinh_moi=B, so_chieu_cu=int(PA.shape[1]), so_chieu_moi=int(PB.shape[1]), giay_ma_hoa_lai_ca_kho=giayB,
                          A_hoi_A=danh_gia(PA, QA), B_hoi_B=danh_gia(PB, QB),
                          hoi_bang_B_tren_kho_A=danh_gia(PA, QB), hoi_bang_A_tren_kho_B=danh_gia(PB, QA),
                          cos_top1_dung_cap=round(statistics.mean(float((PA @ QA[i]).max()) for i in range(len(HOI))), 3),
                          cos_top1_lech_cap=round(statistics.mean(float((PA @ QB[i]).max()) for i in range(len(HOI))), 3))
print("doi mo hinh:", res["doi_mo_hinh"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
