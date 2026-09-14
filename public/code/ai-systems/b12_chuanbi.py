"""B12 chang 0 - tang truy hoi, lam MOT LAN cho moi chang: cat doan, ma hoa bang e5 tren GPU,
tim chinh xac top-3 cho 32 cau hoi (28 co nhan + 4 ngoai kho). Do rieng do tre tang truy hoi."""
import glob, json, os, sys, time, statistics
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b12_chung as C
GOC = "/media/ai/SSD/vietpv/booksys/sach"
CHUONG = ["Foundation", "Machine Learning", "Deep Learning", "Computer Vision", "Generative AI", "AI Systems"]
doan = []
for ch in CHUONG:
    for f in sorted(glob.glob(os.path.join(GOC, ch, "*.md"))):
        t = open(f, encoding="utf-8").read(); i = 0
        while i < len(t):
            d = t[i:i + 800]
            if len(d) > 150: doan.append((ch, os.path.basename(f), d))
            i += 600
import importlib.util
sp = importlib.util.spec_from_file_location("rag", os.path.join(GOC, "Generative AI/code/rag_sach.py"))
rag = importlib.util.module_from_spec(sp); sp.loader.exec_module(rag)
HOI = [(q, ch, so, False) for q, ch, so in rag.CAU_HOI_CO_NHAN] + [
    ("Pha nạp prompt và pha sinh token khác nhau thế nào?", "AI Systems", "01", False),
    ("TTFT và TPOT là gì?", "AI Systems", "01", False),
    ("KV cache tốn bao nhiêu byte cho mỗi token?", "AI Systems", "02", False),
    ("Grouped-query attention giảm bộ nhớ đệm như thế nào?", "AI Systems", "02", False),
    ("Vì sao mô hình 7B không vừa card 12 GB?", "AI Systems", "03", False),
    ("memory_allocated và memory_reserved khác nhau thế nào?", "AI Systems", "03", False),
    ("Gộp lô tĩnh lãng phí ở chỗ nào?", "AI Systems", "05", False),
    ("Tải đóng và tải mở khác nhau thế nào?", "AI Systems", "07", False)] + [(q, None, None, True) for q in C.NGOAI_KHO]
M = "intfloat/multilingual-e5-small"
tk = AutoTokenizer.from_pretrained(M); md = AutoModel.from_pretrained(M).to("cuda").eval()
@torch.no_grad()
def ma(xs, tt):
    out = []
    for k in range(0, len(xs), 64):
        b = tk([tt + x for x in xs[k:k + 64]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = md(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1))
    return torch.cat(out)
t = time.perf_counter(); P = ma([d for _, _, d in doan], "passage: "); torch.cuda.synchronize(); g_kho = time.perf_counter() - t
tre = []; muc = []
for q, ch, so, ngoai in HOI:
    t = time.perf_counter(); v = ma([q], "query: ")[0]; s = P @ v; top = torch.topk(s, 3).indices.tolist()
    torch.cuda.synchronize(); tre.append(time.perf_counter() - t)
    trung = None if ngoai else any(doan[j][0] == ch and doan[j][1].startswith(so + ".") for j in top)
    muc.append(dict(hoi=q, ngoai_kho=ngoai, doan=[doan[j][2] for j in top], nguon=[f"{doan[j][0]}/{doan[j][1]}" for j in top],
                    cos_dau=round(float(s[top[0]]), 4), article_hit3=trung))
res = dict(so_doan=len(doan), giay_ma_hoa_kho_gpu=round(g_kho, 2), so_cau=len(muc),
           article_hit3=f"{sum(m['article_hit3'] for m in muc if not m['ngoai_kho'])}/{sum(not m['ngoai_kho'] for m in muc)}",
           tre_truy_hoi_ms_p50=round(statistics.median(tre) * 1e3, 2), tre_truy_hoi_ms_p95=round(sorted(tre)[int(len(tre) * .95)] * 1e3, 2),
           cos_dau_trong_kho_tb=round(statistics.mean(m["cos_dau"] for m in muc if not m["ngoai_kho"]), 4),
           cos_dau_ngoai_kho_tb=round(statistics.mean(m["cos_dau"] for m in muc if m["ngoai_kho"]), 4), muc=muc)
json.dump(res, open(C.NGU_CANH, "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in res.items() if k != "muc"}, flush=True)
