"""B12 - client cho cac chang B..F. Dung:
  python b12_client.py noitiep <chang>                    # tuan tu, tung cau mot
  python b12_client.py mo <chang> <lambda> <giay> [K]     # tai mo Poisson; K = gioi han so yeu cau dang do (tu choi neu vuot)
Luon doc theo luong de do duoc ca TTFT (thoi diem thay chu dau khi CO luong) va E2E
(= thoi diem thay chu dau khi KHONG luong - bai 08 da do: engine chay y het o hai che do)."""
import asyncio, json, os, random, statistics, subprocess, sys, time, httpx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b12_chung as C
URL = "http://127.0.0.1:8011/v1/chat/completions"; MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"; MAXT = 200
CHE, CHANG = sys.argv[1], sys.argv[2]
muc = C.nap()["muc"]
def pv(xs, q): xs = sorted(xs); return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3) if xs else None
async def mot(cl, m):
    body = dict(model=MODEL, messages=C.tin_nhan(m), max_tokens=MAXT, temperature=0, stream=True)
    t0 = time.perf_counter(); ttft = None; txt = []
    async with cl.stream("POST", URL, json=body) as r:
        async for dong in r.aiter_lines():
            if dong.startswith("data: ") and dong != "data: [DONE]":
                j = json.loads(dong[6:])
                if j["choices"] and j["choices"][0]["delta"].get("content"):
                    if ttft is None: ttft = time.perf_counter() - t0
                    txt.append(j["choices"][0]["delta"]["content"])
    return ttft, time.perf_counter() - t0, "".join(txt)
async def main():
    lim = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
    async with httpx.AsyncClient(timeout=1200, limits=lim) as cl:
        await mot(cl, muc[0])
        kq, tu_choi = [], 0; t0 = time.perf_counter()
        if CHE == "noitiep":
            for i, m in enumerate(muc): kq.append((i, *(await mot(cl, m))))
            lam = None
        else:
            lam, giay = float(sys.argv[3]), float(sys.argv[4]); K = int(sys.argv[5]) if len(sys.argv) > 5 else None
            random.seed(3); dang = [0]; viec = []; i = 0
            async def chay(i):
                dang[0] += 1
                try: kq.append((i % len(muc), *(await mot(cl, muc[i % len(muc)]))))
                finally: dang[0] -= 1
            while time.perf_counter() - t0 < giay:
                if K is not None and dang[0] >= K: tu_choi += 1
                else: viec.append(asyncio.create_task(chay(i)))
                i += 1; await asyncio.sleep(random.expovariate(lam))
            await asyncio.gather(*viec)
        tong = time.perf_counter() - t0
    dau_ra = {}
    for i, _, _, txt in kq: dau_ra.setdefault(i, txt)
    ttft = [k[1] for k in kq if k[1] is not None]; e2e = [k[2] for k in kq]
    vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "1"],
                          capture_output=True, text=True).stdout.strip()
    res = dict(chang=CHANG, che_do=CHE, lambda_dat_ra=lam, gioi_han=(int(sys.argv[5]) if CHE == "mo" and len(sys.argv) > 5 else None),
               so_yeu_cau_xong=len(kq), so_bi_tu_choi=tu_choi, giay=round(tong, 2), req_s=round(len(kq) / tong, 3),
               thay_chu_dau_co_luong_p50=pv(ttft, .5), thay_chu_dau_co_luong_p95=pv(ttft, .95),
               e2e_p50=pv(e2e, .5), e2e_p95=pv(e2e, .95), vram_gpu1_MiB=vram,
               so_cau_co_dau_ra=len(dau_ra))
    if len(dau_ra) == len(muc):
        ds = [dau_ra[i] for i in range(len(muc))]
        res["chat_luong"] = C.cham(muc, ds); res["dau_ra"] = ds
    json.dump(res, open(f"/media/ai/SSD/vietpv/booksys/ket-qua/b12-{CHANG}.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in res.items() if k != "dau_ra"}, flush=True)
asyncio.run(main())
