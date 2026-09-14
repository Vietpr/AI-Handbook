"""B11 - moi loai su co de lai DAU VET tren dai luong nao? Bon tinh huong tren cung may chu:
binh thuong, qua tai, prompt dai, dau ra dai. Doc /metrics cua vLLM truoc/sau moi tinh huong,
va lay mau cac dai luong tuc thoi (so yeu cau dang chay/dang cho, muc dung KV cache)."""
import asyncio, json, random, time, httpx
BASE = "http://127.0.0.1:8011"; URL = BASE + "/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
OUT = "/media/ai/SSD/vietpv/booksys/ket-qua/b11.json"
NGAN = "Viết một đoạn văn ngắn về lịch sử của máy tính cá nhân."
# prompt dai dung theo SO TOKEN do bang tokenizer (lan truoc nhan 120 lan -> vuot 4.096 token -> may chu tra 400)
from transformers import AutoTokenizer
_tk = AutoTokenizer.from_pretrained(MODEL)
_dk = "Điều khoản về bảo mật dữ liệu: nhân viên phải mã hóa thiết bị, không chia sẻ mật khẩu, báo cáo sự cố trong 24 giờ, và tham gia đào tạo định kỳ. "
_n = 1
while len(_tk("Dưới đây là tài liệu nội bộ. " + _dk * (_n + 1)).input_ids) < 2900: _n += 1
DAI = "Dưới đây là tài liệu nội bộ. " + _dk * _n + "\nTóm tắt tài liệu trên trong một câu."
SO_TOKEN_DAI = len(_tk(DAI).input_ids)
TINH_HUONG = {
    "binh_thuong": dict(lam=2, giay=40, prompt=NGAN, max_tok=128),
    "qua_tai":     dict(lam=14, giay=30, prompt=NGAN, max_tok=128),
    "prompt_dai":  dict(lam=1, giay=40, prompt=DAI, max_tok=64),
    "dau_ra_dai":  dict(lam=1, giay=40, prompt=NGAN, max_tok=1024),
}
def doc(txt):
    d = {}
    for dong in txt.splitlines():
        if dong.startswith("vllm:"):
            ten = dong.split("{")[0].split(" ")[0]
            try: d[ten] = d.get(ten, 0.0) + float(dong.rsplit(" ", 1)[1])
            except ValueError: pass
    return d
def tb(a, b, g):
    c = b.get(g + "_count", 0) - a.get(g + "_count", 0)
    return round((b.get(g + "_sum", 0) - a.get(g + "_sum", 0)) / c, 4) if c else None
LOI = {}
async def mot(cl, prompt, max_tok):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=max_tok,
                temperature=0, ignore_eos=True)
    try:
        r = await cl.post(URL, json=body)
        if r.status_code != 200: LOI[r.status_code] = LOI.get(r.status_code, 0) + 1
    except Exception as e:          # mot yeu cau hong khong duoc lam chet ca phep do
        LOI[type(e).__name__] = LOI.get(type(e).__name__, 0) + 1
async def lay_mau(cl, mau, dung):
    while not dung.is_set():
        d = doc((await cl.get(BASE + "/metrics")).text)
        mau.append({k: v for k, v in d.items() if "num_requests" in k or "cache_usage" in k})
        await asyncio.sleep(0.5)
async def main():
    res = dict(model=MODEL, luc=time.strftime("%Y-%m-%d %H:%M:%S"), token_prompt_dai=SO_TOKEN_DAI, tinh_huong={})
    print("prompt dai:", SO_TOKEN_DAI, "token", flush=True)
    async with httpx.AsyncClient(timeout=1200, limits=httpx.Limits(max_connections=1024)) as cl:
        await mot(cl, NGAN, 8)
        for ten, th in TINH_HUONG.items():
            random.seed(1); LOI.clear(); a = doc((await cl.get(BASE + "/metrics")).text)
            mau, dung = [], asyncio.Event(); tm = asyncio.create_task(lay_mau(cl, mau, dung))
            viec, t0 = [], time.perf_counter()
            while time.perf_counter() - t0 < th["giay"]:
                viec.append(asyncio.create_task(mot(cl, th["prompt"], th["max_tok"])))
                await asyncio.sleep(random.expovariate(th["lam"]))
            await asyncio.gather(*viec); dung.set(); await tm
            b = doc((await cl.get(BASE + "/metrics")).text)
            def cao(tu): return max((sum(v for k, v in m.items() if tu in k) for m in mau), default=None)
            r = dict(so_yeu_cau=len(viec),
                     ttft=tb(a, b, "vllm:time_to_first_token_seconds"),
                     cho_hang_doi=tb(a, b, "vllm:request_queue_time_seconds"),
                     nap_prompt=tb(a, b, "vllm:request_prefill_time_seconds"),
                     sinh_token=tb(a, b, "vllm:request_decode_time_seconds"),
                     giua_hai_token=tb(a, b, "vllm:inter_token_latency_seconds"),
                     e2e=tb(a, b, "vllm:e2e_request_latency_seconds"),
                     token_vao_tb=tb(a, b, "vllm:request_prompt_tokens"),
                     token_ra_tb=tb(a, b, "vllm:request_generation_tokens"),
                     dang_cho_cao_nhat=cao("num_requests_waiting"),
                     dang_chay_cao_nhat=cao("num_requests_running"),
                     kv_dung_cao_nhat=cao("cache_usage"),
                     loi=dict(LOI), ten_gauge=sorted({k for m in mau[:1] for k in m}))
            res["tinh_huong"][ten] = r; print(ten, r, flush=True)
            await asyncio.sleep(3)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("da ghi", OUT, flush=True)
asyncio.run(main())
