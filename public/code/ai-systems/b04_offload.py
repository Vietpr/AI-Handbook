"""B04 - loi thoat thu hai: giu mot phan lop tren GPU, phan con lai nam o RAM
va duoc CHEP QUA PCIe moi lan can. Ban cai dat toi gian, viet tay de thay ro co che.
Truoc khi chay, do bang thong PCIe de DU DOAN toc do, roi moi do toc do that."""
import json, statistics, subprocess, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b04-offload.json"
NGAN_SACH_GIB = 9.0          # phan VRAM danh cho trong so thuong tru
DU_TRU_GIB = 1.0             # chua cho KV cache va hoat hoa
# may dung chung: neu RAM trong khong du cho 15,2 GB trong so thi DUNG, dung de he dieu hanh
# phai giet tien trinh cua nguoi khac
def ram_trong_gib():
    for d in open("/proc/meminfo"):
        if d.startswith("MemAvailable:"): return int(d.split()[1]) / 2**20
trong = ram_trong_gib()
print("RAM trong:", round(trong, 1), "GiB", flush=True)
if trong < 17.5:
    raise SystemExit(f"DUNG: RAM trong chi {trong:.1f} GiB, can it nhat 17,5 GiB de nap 15,2 GB an toan")

res = dict(model=MODEL, luc=time.strftime("%Y-%m-%d %H:%M:%S"), ram_trong_GiB=round(trong, 1),
           gpu=subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())

# ---- 1. bang thong PCIe, RAM -> GPU ----
def bw(pin):
    x = torch.empty(512 * 2**20, dtype=torch.uint8)
    if pin: x = x.pin_memory()
    y = torch.empty_like(x, device="cuda")
    for _ in range(3): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(); return 10 * x.numel() / (time.perf_counter() - t) / 1e9
res["pcie_GBps"] = dict(bo_nho_ghim=round(bw(True), 2), bo_nho_thuong=round(bw(False), 2))
print("PCIe:", res["pcie_GBps"], flush=True)

# ---- 2. nap mo hinh len RAM, chon bao nhieu lop o lai GPU ----
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
lops = mdl.model.layers
b_lop = sum(p.numel() * p.element_size() for p in lops[0].parameters())
b_nen = sum(p.numel() * p.element_size() for m in (mdl.model.embed_tokens, mdl.model.norm, mdl.lm_head)
            for p in m.parameters())
K = int((NGAN_SACH_GIB * 2**30 - b_nen - DU_TRU_GIB * 2**30) // b_lop)
K = max(0, min(K, len(lops)))
so_chep = len(lops) - K
res["bo_tri"] = dict(so_lop=len(lops), lop_thuong_tru=K, lop_phai_chep=so_chep,
                     GB_moi_lop=round(b_lop / 1e9, 3), GB_nen=round(b_nen / 1e9, 3),
                     GB_chep_moi_token=round(so_chep * b_lop / 1e9, 2))
print("bo tri:", res["bo_tri"], flush=True)

for m in (mdl.model.embed_tokens, mdl.model.norm, mdl.lm_head, mdl.model.rotary_emb):
    m.to("cuda")
for l in lops[:K]: l.to("cuda")

class Chep:
    """Giu ban tren RAM (ghim), chep len GPU truoc khi lop chay, bo ban GPU sau khi chay."""
    def __init__(self, lop):
        self.cpu = {n: p.data.pin_memory() for n, p in lop.named_parameters()}
        for n, p in lop.named_parameters(): p.data = self.cpu[n]
        lop.register_forward_pre_hook(self.truoc)
        lop.register_forward_hook(self.sau)
    def truoc(self, mod, args):
        for n, p in mod.named_parameters(): p.data = self.cpu[n].to("cuda", non_blocking=True)
    def sau(self, mod, args, out):
        for n, p in mod.named_parameters(): p.data = self.cpu[n]
        return out
for l in lops[K:]: Chep(l)
torch.cuda.synchronize()
res["vram_sau_bo_tri_GiB"] = round(torch.cuda.memory_allocated() / 2**30, 2)
print("VRAM thuong tru:", res["vram_sau_bo_tri_GiB"], "GiB", flush=True)

# ---- 3. du doan roi do ----
du_doan = res["pcie_GBps"]["bo_nho_ghim"] / res["bo_tri"]["GB_chep_moi_token"] if so_chep else None
res["du_doan_tok_s_tran_tren"] = round(du_doan, 2) if du_doan else None
print("du doan tran tren:", res["du_doan_tok_s_tran_tren"], "tok/s", flush=True)

ids = tok("Học máy là ngành nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng.",
          return_tensors="pt").input_ids.to("cuda")
@torch.no_grad()
def chay(so_token):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o = mdl(ids, use_cache=True); nxt = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(); ttft = time.perf_counter() - t0
    cache, itl = o.past_key_values, []
    for _ in range(so_token - 1):
        t1 = time.perf_counter()
        o = mdl(nxt, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1:].argmax(-1); cache = o.past_key_values
        torch.cuda.synchronize(); itl.append(time.perf_counter() - t1)
    return ttft, itl
chay(4)
ttft, itl = chay(24)
res["do_that"] = dict(ttft_s=round(ttft, 3), tpot_p50_ms=round(statistics.median(itl) * 1e3, 1),
                      tok_s=round(1 / statistics.median(itl), 2),
                      vram_dinh_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
print("do that:", res["do_that"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
