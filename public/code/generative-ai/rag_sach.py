"""
RAG end-to-end trên 51 bài của bốn chương đầu — chạy được từ đầu tới cuối.

    python3 rag_sach.py --hoi "IoU được tính như thế nào?"
    python3 rag_sach.py --danh-gia          # chấm article hit@k trên 20 câu có nhãn

Mặc định tệp tự tìm thư mục sách ở hai cấp trên chính nó. Nếu bạn để sách ở
chỗ khác thì truyền đường dẫn vào:

    python3 rag_sach.py --thu-muc-sach /duong/dan/toi/Book --hoi "..."

Cần: transformers, torch. Lần đầu chạy sẽ tải hai mô hình (~1,5 GB).
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# thư mục sách: cờ dòng lệnh > biến môi trường > hai cấp trên chính tệp này
MAC_DINH = Path(__file__).resolve().parents[2]
GOC = os.environ.get("BOOK_DIR", str(MAC_DINH))
CHUONG = ("Foundation", "Machine Learning", "Deep Learning", "Computer Vision")
EMB = "intfloat/multilingual-e5-small"
LM = "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_num_threads(4)

# ---------------------------------------------------------------- 1. nạp + cắt đoạn
def nap_va_cat(chunk=800, overlap=200):
    docs = []
    for ch in CHUONG:
        for f in sorted(glob.glob(os.path.join(GOC, ch, "*.md"))):
            t = open(f, encoding="utf-8").read()
            bai = os.path.basename(f)
            i = 0
            while i < len(t):
                doan = t[i:i + chunk]
                if len(doan) > 150:
                    docs.append(dict(chuong=ch, bai=bai, noi_dung=doan))
                i += chunk - overlap
    if not docs:
        sys.exit(f"Không tìm thấy bài nào trong {GOC}.\n"
                 f"Truyền --thu-muc-sach /duong/dan/toi/Book cho đúng.")
    return docs

# ---------------------------------------------------------------- 2. mã hóa
class BoMaHoa:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.mdl = AutoModel.from_pretrained(EMB).eval()

    @torch.no_grad()
    def __call__(self, texts, tien_to):
        ra = []
        for i in range(0, len(texts), 16):
            e = self.tok([tien_to + t for t in texts[i:i + 16]], padding=True,
                         truncation=True, max_length=512, return_tensors="pt")
            o = self.mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            ra.append(F.normalize((o * m).sum(1) / m.sum(1), dim=1))
        return torch.cat(ra)

# ---------------------------------------------------------------- 3. truy hồi
def truy_hoi(kho_vec, docs, enc, cau_hoi, k=3):
    qv = enc([cau_hoi], "query: ")[0]
    diem = kho_vec @ qv
    idx = diem.topk(k).indices.tolist()
    return [(docs[i], float(diem[i])) for i in idx]

# ---------------------------------------------------------------- 4. sinh
def sinh(tok, lm, cau_hoi, ket_qua):
    ctx = "\n\n".join(f"[{i+1}] ({d['bai']}) {d['noi_dung'][:700]}"
                      for i, (d, _) in enumerate(ket_qua))
    p = ("Dựa CHỈ vào các đoạn trích dưới đây, trả lời câu hỏi. "
         "Ghi số nguồn [n] sau câu trả lời. "
         "Nếu các đoạn trích không chứa câu trả lời, hãy nói rõ là không tìm thấy.\n\n"
         f"{ctx}\n\nCâu hỏi: {cau_hoi}\nTrả lời:")
    text = tok.apply_chat_template([{"role": "user", "content": p}],
                                   add_generation_prompt=True, tokenize=False)
    e = tok(text, return_tensors="pt", truncation=True, max_length=3000)
    with torch.no_grad():
        o = lm.generate(**e, max_new_tokens=120, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ---------------------------------------------------------------- 5. kiểm trích dẫn bằng luật
def kiem_trich_dan(tra_loi, so_nguon):
    duoc_dan = {int(x) for x in re.findall(r"\[(\d+)\]", tra_loi)}
    hop_le = duoc_dan <= set(range(1, so_nguon + 1))
    return dict(co_trich_dan=bool(duoc_dan), hop_le=hop_le,
                ngoai_pham_vi=sorted(duoc_dan - set(range(1, so_nguon + 1))))

# ---------------------------------------------------------------- 6. bộ câu hỏi có nhãn
CAU_HOI_CO_NHAN = [
    ("Vì sao phải chia dữ liệu thành ba tập train, validation và test?", "Foundation", "04"),
    ("Overfitting là gì và dấu hiệu nhận biết?", "Foundation", "14"),
    ("Ridge và Lasso khác nhau ở chỗ nào?", "Machine Learning", "04"),
    ("k-means hoạt động thế nào?", "Machine Learning", "09"),
    ("PCA dùng để làm gì?", "Machine Learning", "10"),
    ("Vì sao mạng tuyến tính không giải được XOR?", "Deep Learning", "01"),
    ("ReLU chết là hiện tượng gì?", "Deep Learning", "02"),
    ("Backward pass tốn chi phí bao nhiêu so với forward?", "Deep Learning", "04"),
    ("Dropout hoạt động ra sao?", "Deep Learning", "07"),
    ("Vanishing gradient xảy ra vì sao?", "Deep Learning", "08"),
    ("BatchNorm ở chế độ eval dùng thống kê nào?", "Deep Learning", "08"),
    ("CNN khai thác giả định gì về ảnh?", "Deep Learning", "09"),
    ("Transfer learning có mấy chiến lược?", "Deep Learning", "11"),
    ("Thang giá trị 0-255 gây hậu quả gì cho mô hình?", "Computer Vision", "01"),
    ("Trường tiếp nhận được tính thế nào?", "Computer Vision", "02"),
    ("Kết nối tắt giúp gì cho mạng sâu?", "Computer Vision", "03"),
    ("Grad-CAM được cài đặt ra sao?", "Computer Vision", "04"),
    ("IoU tính như thế nào?", "Computer Vision", "07"),
    ("NMS là thuật toán gì?", "Computer Vision", "07"),
    ("ViT cắt ảnh thành gì?", "Computer Vision", "10"),
]

def danh_gia_truy_hoi(enc):
    """article hit@k — nhãn chỉ ở mức BÀI, không phải mức đoạn."""
    print(f"{'cấu hình':<26}{'số đoạn':>9}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}")
    qv = enc([q for q, _, _ in CAU_HOI_CO_NHAN], "query: ")
    for chunk, ov in ((800, 0), (800, 200), (1500, 200), (400, 100)):
        docs = nap_va_cat(chunk, ov)
        dv = enc([d["noi_dung"] for d in docs], "passage: ")
        hit = {1: 0, 3: 0, 5: 0}
        for (q, ch, so), v in zip(CAU_HOI_CO_NHAN, qv):
            top = (dv @ v).topk(5).indices.tolist()
            for k in (1, 3, 5):
                if any(docs[j]["chuong"] == ch and docs[j]["bai"].startswith(so)
                       for j in top[:k]):
                    hit[k] += 1
        n = len(CAU_HOI_CO_NHAN)
        print(f"{f'{chunk} ký tự, chồng {ov}':<26}{len(docs):>9}"
              f"{hit[1]/n:>8.2f}{hit[3]/n:>8.2f}{hit[5]/n:>8.2f}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoi", type=str, default=None)
    ap.add_argument("--danh-gia", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--thu-muc-sach", type=str, default=None,
                    help="thư mục chứa các thư mục chương (mặc định: hai cấp trên tệp này)")
    a = ap.parse_args()

    global GOC
    if a.thu_muc_sach:
        GOC = a.thu_muc_sach

    enc = BoMaHoa()
    if a.danh_gia:
        danh_gia_truy_hoi(enc)
        return

    cau_hoi = a.hoi or "IoU được tính như thế nào?"
    docs = nap_va_cat()
    print(f"kho: {len(docs)} đoạn từ {len(CHUONG)} chương", flush=True)
    kho_vec = enc([d["noi_dung"] for d in docs], "passage: ")
    kq = truy_hoi(kho_vec, docs, enc, cau_hoi, a.k)

    print(f"\ncâu hỏi: {cau_hoi}\nnguồn truy hồi:")
    for i, (d, s) in enumerate(kq, 1):
        print(f"  [{i}] {d['chuong']}/{d['bai'][:40]}  (cosine {s:.4f})")

    tok = AutoTokenizer.from_pretrained(LM)
    lm = AutoModelForCausalLM.from_pretrained(LM, dtype=torch.float32).eval()
    tra_loi = sinh(tok, lm, cau_hoi, kq)
    print(f"\ntrả lời:\n{tra_loi}")
    print(f"\nkiểm trích dẫn bằng luật: {kiem_trich_dan(tra_loi, len(kq))}")

if __name__ == "__main__":
    main()
