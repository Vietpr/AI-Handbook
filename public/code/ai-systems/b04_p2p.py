"""B04 - truoc khi do tensor parallel: hai GPU noi chuyen voi nhau qua duong nao, nhanh bao nhieu."""
import json, subprocess, time, torch
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b04-p2p.json"
sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
res = dict(luc=time.strftime("%Y-%m-%d %H:%M:%S"),
           topo=sh("nvidia-smi", "topo", "-m"), p2p_doc=sh("nvidia-smi", "topo", "-p2p", "r"),
           gpu=sh("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"),
           tien_trinh_khac=sh("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"),
           truy_cap_ngang_hang=torch.cuda.can_device_access_peer(0, 1))
def chep(n_mib=512):
    x = torch.empty(n_mib * 2**20, dtype=torch.uint8, device="cuda:0")
    y = torch.empty_like(x, device="cuda:1")
    for _ in range(3): y.copy_(x)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1); t = time.perf_counter()
    for _ in range(10): y.copy_(x)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    return 10 * x.numel() / (time.perf_counter() - t) / 1e9
res["GPU0_sang_GPU1_GBps"] = round(chep(), 2)
print(json.dumps({k: v for k, v in res.items() if k not in ("topo",)}, ensure_ascii=False, indent=1))
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
