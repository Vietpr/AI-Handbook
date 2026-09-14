"""B02 phu - kiem hai gia thuyet: (a) delta bo nho bi logits chi phoi, (b) ty le cong viec khi bo cache."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(M)
mdl=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
cfg=mdl.config
kv_tok = 2*cfg.num_hidden_layers*cfg.num_key_value_heads*(cfg.hidden_size//cfg.num_attention_heads)*2
lg_tok = cfg.vocab_size*2
print("vocab:", cfg.vocab_size)
print("KV     :", kv_tok, "byte/token =", round(kv_tok/1024,1), "KiB")
print("logits :", lg_tok, "byte/token =", round(lg_tok/1024,1), "KiB  ->", round(lg_tok/kv_tok,1), "lan KV")
res=dict(vocab=cfg.vocab_size, kv_byte_token=kv_tok, logits_byte_token=lg_tok,
         ty_le_logits_tren_kv=round(lg_tok/kv_tok,2), do={})
nen="Hoc may la nganh nghien cuu cac thuat toan cho phep may tinh cai thien hieu nang. "
for T in (2048, 8192):
    ids=tok(nen*(T//14+3), return_tensors="pt").input_ids[:,:T].to("cuda")
    torch.cuda.empty_cache(); truoc=torch.cuda.memory_allocated()
    with torch.no_grad(): o=mdl(ids, use_cache=True)
    sau=torch.cuda.memory_allocated()
    lg=o.logits.numel()*o.logits.element_size()
    kvs=sum(t.numel()*t.element_size() for lay in o.past_key_values.layers
            for t in (lay.keys, lay.values) if torch.is_tensor(t))
    res["do"][T]=dict(logits_shape=str(tuple(o.logits.shape)), logits_byte=lg, logits_dtype=str(o.logits.dtype),
                      kv_byte=kvs, delta=sau-truoc, logits_cong_kv=lg+kvs,
                      logits_chiem_phan_tram=round(100*lg/(sau-truoc),1))
    print(T, res["do"][T], flush=True)
    del o; torch.cuda.empty_cache()
# ty le CONG VIEC khi bo cache (so token-vi tri phai xu ly)
P=128
for N in (64,128,256,512):
    co=P+(N-1); khong=sum(P+i for i in range(N))
    res.setdefault("cong_viec",{})[N]=dict(co_cache=co, khong_cache=khong, ty_le=round(khong/co,1))
    print("cong viec N=",N,res["cong_viec"][N], flush=True)
json.dump(res, open("/media/ai/SSD/vietpv/booksys/ket-qua/b02b.json","w"), indent=1, ensure_ascii=False)
print("da ghi")
