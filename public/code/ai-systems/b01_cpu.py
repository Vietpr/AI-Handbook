"""B01 phu - cung phep do tren CPU de doi chieu hinh dang."""
import json, statistics, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL="Qwen/Qwen2.5-1.5B-Instruct"; OUT="/media/ai/SSD/vietpv/booksys/ket-qua/b01-cpu.json"
torch.set_num_threads(20)
tok=AutoTokenizer.from_pretrained(MODEL)
mdl=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
byte_ts=sum(p.numel()*p.element_size() for p in mdl.parameters())
print("CPU fp32, trong so", round(byte_ts/1e9,2),"GB | threads", torch.get_num_threads(), flush=True)
nen=("Hoc may la nganh nghien cuu cac thuat toan cho phep may tinh cai thien "
     "hieu nang tren mot nhiem vu thong qua kinh nghiem. ")
SINH=24; ket={}
@torch.no_grad()
def mot_luot(ids):
    t0=time.perf_counter(); out=mdl(ids, use_cache=True)
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True); ttft=time.perf_counter()-t0
    cache=out.past_key_values; itls=[]
    for _ in range(SINH-1):
        t1=time.perf_counter(); o=mdl(nxt, past_key_values=cache, use_cache=True)
        nxt=o.logits[:,-1,:].argmax(-1,keepdim=True); cache=o.past_key_values
        itls.append(time.perf_counter()-t1)
    return ttft, itls, ttft+sum(itls)
for L in (256, 1024):
    ids=tok(nen*(L//18+2), return_tensors="pt").input_ids[:,:L]
    assert ids.shape[1]==L
    mot_luot(ids)
    ts=[];il=[];ee=[]
    for _ in range(2):
        a,b,c=mot_luot(ids); ts.append(a); il+=b; ee.append(c)
    il.sort()
    r=dict(ttft_s=round(statistics.median(ts),3),
           tpot_p50_ms=round(statistics.median(il)*1e3,1),
           e2e_s=round(statistics.median(ee),2),
           nap_prompt_tok_s=round(L/statistics.median(ts),1),
           sinh_tok_s=round(1/statistics.median(il),2))
    r["bang_thong_dat_GBps"]=round(byte_ts*r["sinh_tok_s"]/1e9,1)
    ket[L]=r; print(L, r, flush=True)
json.dump(dict(model=MODEL, dtype="float32", threads=20, byte_trong_so=byte_ts, cpu=ket),
          open(OUT,"w"), indent=1, ensure_ascii=False)
print("da ghi", OUT)
