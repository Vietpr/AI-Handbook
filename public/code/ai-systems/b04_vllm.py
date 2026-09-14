"""B04 - do mot cau hinh trien khai 7B bang vLLM: bo nho, toc do, chat luong.
Dung:  python b04_vllm.py <ten_cau_hinh> <model> <tp> [quant]"""
import json, os, re, subprocess, sys, time, random, unicodedata
TEN, MODEL, TP = sys.argv[1], sys.argv[2], int(sys.argv[3])
QUANT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "none" else None
GMU = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
OUT = f"/media/ai/SSD/vietpv/booksys/ket-qua/b04-{TEN}.json"

def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

res = dict(cau_hinh=TEN, model=MODEL, tp=TP, quant=QUANT, gpu_memory_utilization=GMU, luc=time.strftime("%Y-%m-%d %H:%M:%S"),
           gpu_truoc=smi())
print("GPU truoc:", res["gpu_truoc"], flush=True)

from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=MODEL, tensor_parallel_size=TP, quantization=QUANT, dtype="auto",
          gpu_memory_utilization=GMU, max_model_len=4096, seed=0, enforce_eager=False)
res["giay_khoi_dong"] = round(time.time() - t0, 1)
res["gpu_sau_khi_nap"] = smi()
print("nap xong", res["giay_khoi_dong"], "s |", res["gpu_sau_khi_nap"], flush=True)

tok = llm.get_tokenizer()
def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True)

# ---- toc do sinh o batch 1 ----
sp = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
p1 = chat("Viet mot doan van dai ve lich su may tinh.")
llm.generate([p1], sp, use_tqdm=False)
t = time.perf_counter(); o = llm.generate([p1], sp, use_tqdm=False); dt = time.perf_counter() - t
res["batch1"] = dict(token_ra=len(o[0].outputs[0].token_ids), giay=round(dt, 3),
                     tok_s=round(len(o[0].outputs[0].token_ids) / dt, 1))
print("batch1:", res["batch1"], flush=True)

# ---- thong luong o batch 32 ----
ps = [chat(f"Viet mot doan van ve chu de so {i}.") for i in range(32)]
t = time.perf_counter(); os_ = llm.generate(ps, sp, use_tqdm=False); dt = time.perf_counter() - t
n = sum(len(x.outputs[0].token_ids) for x in os_)
res["batch32"] = dict(token_ra=n, giay=round(dt, 3), tok_s=round(n / dt, 1))
print("batch32:", res["batch32"], flush=True)

# ---- chat luong: bo viec co dap an ----
random.seed(7)
viec = []
for _ in range(30):
    a, b = random.randint(100, 999), random.randint(100, 999)
    viec.append((f"Tính {a} + {b}. Chỉ trả lời bằng một số, không giải thích.", str(a + b)))
for _ in range(20):
    a, b = random.randint(12, 99), random.randint(12, 99)
    viec.append((f"Tính {a} × {b}. Chỉ trả lời bằng một số, không giải thích.", str(a * b)))
su_that = [("Thủ đô của Nhật Bản là thành phố nào? Trả lời một từ.", "tokyo"),
           ("Thủ đô của Pháp là thành phố nào? Trả lời một từ.", "paris"),
           ("Thủ đô của Úc là thành phố nào? Trả lời một từ.", "canberra"),
           ("Thủ đô của Canada là thành phố nào? Trả lời một từ.", "ottawa"),
           ("Thủ đô của Brazil là thành phố nào? Trả lời một từ.", "brasilia"),
           ("Nước sôi ở bao nhiêu độ C ở mực nước biển? Chỉ trả lời số.", "100"),
           ("Một tuần có bao nhiêu ngày? Chỉ trả lời số.", "7"),
           ("Hành tinh lớn nhất Hệ Mặt Trời tên là gì? Trả lời một từ.", "moc tinh|jupiter"),
           ("Ký hiệu hóa học của vàng là gì? Chỉ trả lời ký hiệu.", "au"),
           ("Tác giả của Truyện Kiều là ai?", "nguyen du")]
viec += su_that
spq = SamplingParams(temperature=0, max_tokens=24)
outs = llm.generate([chat(q) for q, _ in viec], spq, use_tqdm=False)
def chuan(x):
    x = unicodedata.normalize("NFD", x.lower()).replace("đ", "d")
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", x.replace(",", "").replace(".", " "))
def dung(tra_loi, dap):
    return any(chuan(d) in chuan(tra_loi) for d in dap.split("|"))
ket = [dung(o.outputs[0].text, d) for o, (_, d) in zip(outs, viec)]
res["chat_luong"] = dict(tong=len(ket), dung=sum(ket),
                         cong=sum(ket[:30]), nhan=sum(ket[30:50]), su_that=sum(ket[50:]),
                         tat_ca=[dict(hoi=viec[i][0], dap=viec[i][1], tra_loi=outs[i].outputs[0].text, dung=k)
                                 for i, k in enumerate(ket)])
print("chat luong:", {k: v for k, v in res["chat_luong"].items() if k != "tat_ca"}, flush=True)
res["gpu_cuoi"] = smi()
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("da ghi", OUT, flush=True)
