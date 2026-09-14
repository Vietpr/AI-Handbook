"""B01 phu - do TRAN thuc te cua chinh GPU nay: tinh toan bf16 va bang thong bo nho."""
import json, time, torch
OUT="/media/ai/SSD/vietpv/booksys/ket-qua/b01-tran.json"
d="cuda"; res={}
p=torch.cuda.get_device_properties(0)
res["thiet_bi"]=dict(ten=p.name, sm=p.multi_processor_count, vram_GB=round(p.total_memory/1e9,2),
                     cc=f"{p.major}.{p.minor}", torch=torch.__version__, cuda=torch.version.cuda)

def do(f, lap=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(lap): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/lap

# --- tran tinh toan: nhan ma tran bf16 lon ---
best=0
for N in (4096, 8192):
    a=torch.randn(N,N,device=d,dtype=torch.bfloat16); b=torch.randn(N,N,device=d,dtype=torch.bfloat16)
    t=do(lambda: a@b, 20); tf=2*N**3/t/1e12
    print(f"GEMM bf16 {N}x{N}: {tf:.2f} TFLOP/s", flush=True); best=max(best,tf)
    del a,b; torch.cuda.empty_cache()
res["tran_tinh_toan_bf16_TFLOPs"]=round(best,2)

# --- tran bang thong: copy tensor lon (doc + ghi) ---
bw=0
for MB in (256, 1024):
    n=MB*1024*1024//2
    x=torch.empty(n,device=d,dtype=torch.bfloat16).normal_(); y=torch.empty_like(x)
    t=do(lambda: y.copy_(x), 50); g=2*x.numel()*2/t/1e9   # doc x + ghi y
    print(f"copy {MB} MB: {g:.1f} GB/s", flush=True); bw=max(bw,g)
    del x,y; torch.cuda.empty_cache()
res["tran_bang_thong_GBps"]=round(bw,1)
json.dump(res, open(OUT,"w"), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False))
