"""B04 - moc CHAT LUONG cua ban bf16 (khong luong tu hoa) tren dung 60 viec da cham cho AWQ.
Ban 7B bf16 khong vua mot card, nen chay bang offload (bai nay, muc 3) va GOP CA 60 CAU MOT LO:
moi lan chep trong so qua PCIe phuc vu ca 60 cau."""
import json, random, re, time, unicodedata, subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = "Qwen/Qwen2.5-7B-Instruct"; OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b04-chatluong-bf16.json"
def ram():
    for d in open("/proc/meminfo"):
        if d.startswith("MemAvailable:"): return int(d.split()[1]) / 2**20
if ram() < 17.5: raise SystemExit(f"DUNG: RAM trong {ram():.1f} GiB")
# ---- dung y het bo viec va bo cham cua b04_vllm.py ----
random.seed(7); viec = []
for _ in range(30):
    a, b = random.randint(100, 999), random.randint(100, 999)
    viec.append((f"Tính {a} + {b}. Chỉ trả lời bằng một số, không giải thích.", str(a + b)))
for _ in range(20):
    a, b = random.randint(12, 99), random.randint(12, 99)
    viec.append((f"Tính {a} × {b}. Chỉ trả lời bằng một số, không giải thích.", str(a * b)))
viec += [("Thủ đô của Nhật Bản là thành phố nào? Trả lời một từ.", "tokyo"),
         ("Thủ đô của Pháp là thành phố nào? Trả lời một từ.", "paris"),
         ("Thủ đô của Úc là thành phố nào? Trả lời một từ.", "canberra"),
         ("Thủ đô của Canada là thành phố nào? Trả lời một từ.", "ottawa"),
         ("Thủ đô của Brazil là thành phố nào? Trả lời một từ.", "brasilia"),
         ("Nước sôi ở bao nhiêu độ C ở mực nước biển? Chỉ trả lời số.", "100"),
         ("Một tuần có bao nhiêu ngày? Chỉ trả lời số.", "7"),
         ("Hành tinh lớn nhất Hệ Mặt Trời tên là gì? Trả lời một từ.", "moc tinh|jupiter"),
         ("Ký hiệu hóa học của vàng là gì? Chỉ trả lời ký hiệu.", "au"),
         ("Tác giả của Truyện Kiều là ai?", "nguyen du")]
def chuan(x):
    x = unicodedata.normalize("NFD", x.lower()).replace("đ", "d")
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", x.replace(",", "").replace(".", " "))
def dung(t, d): return any(chuan(x) in chuan(t) for x in d.split("|"))
# ---- offload: giu 13 lop tren GPU, chep 15 lop con lai moi buoc ----
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
lops = mdl.model.layers; K = 13
for m in (mdl.model.embed_tokens, mdl.model.norm, mdl.lm_head, mdl.model.rotary_emb): m.to("cuda")
for l in lops[:K]: l.to("cuda")
class Chep:
    def __init__(self, lop):
        self.cpu = {n: p.data.pin_memory() for n, p in lop.named_parameters()}
        for n, p in lop.named_parameters(): p.data = self.cpu[n]
        lop.register_forward_pre_hook(self.truoc); lop.register_forward_hook(self.sau)
    def truoc(self, mod, args):
        for n, p in mod.named_parameters(): p.data = self.cpu[n].to("cuda", non_blocking=True)
    def sau(self, mod, args, out):
        for n, p in mod.named_parameters(): p.data = self.cpu[n]
        return out
for l in lops[K:]: Chep(l)
ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in viec]
b = tok(ps, return_tensors="pt", padding=True).to("cuda")
t0 = time.perf_counter()
with torch.no_grad():
    o = mdl.generate(**b, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
dt = time.perf_counter() - t0
tra_loi = tok.batch_decode(o[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
ket = [dung(t, d) for t, (_, d) in zip(tra_loi, viec)]
res = dict(model=MODEL, dtype="bfloat16", cach_chay="offload 13 lop thuong tru, lo 60 cau", giay=round(dt, 1),
           chat_luong=dict(tong=len(ket), dung=sum(ket), cong=sum(ket[:30]), nhan=sum(ket[30:50]), su_that=sum(ket[50:]),
                           tat_ca=[dict(hoi=viec[i][0], dap=viec[i][1], tra_loi=tra_loi[i], dung=k) for i, k in enumerate(ket)]))
try:
    awq = json.load(open("/media/ai/SSD/vietpv/booksys/ket-qua/b04-marlin1.json"))["chat_luong"]["tat_ca"]
    res["giong_het_awq"] = sum(chuan(x["tra_loi"]) == chuan(y["tra_loi"]) for x, y in zip(res["chat_luong"]["tat_ca"], awq))
    res["cau_dung_khac_nhau"] = [dict(hoi=x["hoi"][:40], bf16=x["tra_loi"][:20], awq=y["tra_loi"][:20])
                                 for x, y in zip(res["chat_luong"]["tat_ca"], awq) if x["dung"] != y["dung"]]
except Exception as e: res["giong_het_awq"] = f"loi {e}"
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in res.items() if k != "chat_luong"}, {k: v for k, v in res["chat_luong"].items() if k != "tat_ca"}, flush=True)
