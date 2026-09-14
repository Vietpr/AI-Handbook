"""B01 phu - do TRAN cua CPU giong het cach da do tren GPU."""
import json, time, torch
torch.set_num_threads(4)
def do(f, lap):
    for _ in range(3): f()
    t0=time.perf_counter()
    for _ in range(lap): f()
    return (time.perf_counter()-t0)/lap
best=0
for N in (1024, 2048):
    a=torch.randn(N,N); b=torch.randn(N,N)
    t=do(lambda: a@b, 10); tf=2*N**3/t/1e12
    print(f"GEMM fp32 {N}x{N}: {tf*1000:.1f} GFLOP/s", flush=True); best=max(best,tf)
bw=0
for MB in (128, 512):
    n=MB*1024*1024//4
    x=torch.empty(n).normal_(); y=torch.empty_like(x)
    t=do(lambda: y.copy_(x), 20); g=2*x.numel()*4/t/1e9
    print(f"copy {MB} MB: {g:.1f} GB/s", flush=True); bw=max(bw,g)
r=dict(tran_tinh_toan_fp32_GFLOPs=round(best*1000,1), tran_bang_thong_GBps=round(bw,1),
       diem_can_bang_FLOP_byte=round(best*1e12/(bw*1e9),1), threads=4)
json.dump(r, open(f"{__file__.rsplit('/',1)[0]}/b01-tran-cpu.json","w"), indent=1)
print(json.dumps(r))
