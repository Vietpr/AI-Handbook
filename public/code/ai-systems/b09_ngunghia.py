"""B09 phan B - bo dem theo ngu nghia: lay LAI CAU TRA LOI CU khi cau hoi moi 'gan nghia'.
Moi cau goc co 2 cach noi lai (PHAI trung) va 2 cau gan giong nhung khac dieu kien then chot
(trung la tra nham dap an). Quet nguong cosine de thay danh doi."""
import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
M = "intfloat/multilingual-e5-small"
OUT = __file__.rsplit("/", 1)[0] + "/b09-ngunghia.json"
BO = [
 ("Giá vé vào cổng cho người lớn là bao nhiêu?", ["Người lớn mua vé vào cửa hết bao nhiêu tiền?", "Vé người lớn giá bao nhiêu?"],
  ["Giá vé vào cổng cho trẻ em là bao nhiêu?", "Giá vé vào cổng cho người cao tuổi là bao nhiêu?"]),
 ("Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?", ["Số ngày phép năm của nhân viên chính thức là bao nhiêu?", "Nhân viên chính thức có mấy ngày phép một năm?"],
  ["Nhân viên thử việc được nghỉ phép năm bao nhiêu ngày?", "Thực tập sinh được nghỉ phép năm bao nhiêu ngày?"]),
 ("Văn phòng Hà Nội mở cửa lúc mấy giờ?", ["Mấy giờ thì văn phòng ở Hà Nội mở cửa?", "Giờ mở cửa của văn phòng Hà Nội là mấy giờ?"],
  ["Văn phòng TP.HCM mở cửa lúc mấy giờ?", "Văn phòng Hà Nội đóng cửa lúc mấy giờ?"]),
 ("Doanh thu quý 1 năm 2024 là bao nhiêu?", ["Quý 1/2024 doanh thu đạt bao nhiêu?", "Doanh thu trong quý đầu tiên của năm 2024 là bao nhiêu?"],
  ["Doanh thu quý 1 năm 2025 là bao nhiêu?", "Doanh thu quý 2 năm 2024 là bao nhiêu?"]),
 ("Làm thế nào để đặt lại mật khẩu email công ty?", ["Tôi quên mật khẩu email công ty, làm sao đặt lại?", "Cách lấy lại mật khẩu email công ty?"],
  ["Làm thế nào để đặt lại mật khẩu VPN công ty?", "Làm thế nào để đổi tên hiển thị email công ty?"]),
 ("Có được mang thú cưng vào khách sạn không?", ["Khách sạn có cho phép mang thú cưng không?", "Tôi mang chó mèo vào khách sạn được không?"],
  ["Có được hút thuốc trong khách sạn không?", "Có được mang thú cưng vào nhà hàng không?"]),
 ("Thuốc này có dùng được cho phụ nữ mang thai không?", ["Phụ nữ có thai uống thuốc này được không?", "Bà bầu có dùng được thuốc này không?"],
  ["Thuốc này có dùng được cho trẻ em dưới 2 tuổi không?", "Thuốc này có dùng được cho phụ nữ cho con bú không?"]),
 ("Hạn chót nộp báo cáo thuế năm nay là ngày nào?", ["Năm nay hạn cuối nộp báo cáo thuế là khi nào?", "Báo cáo thuế năm nay phải nộp trước ngày nào?"],
  ["Hạn chót nộp báo cáo thuế năm ngoái là ngày nào?", "Hạn chót nộp tờ khai hải quan năm nay là ngày nào?"]),
 ("Chuyển khoản trên 500 triệu có cần xác thực thêm không?", ["Chuyển trên 500 triệu đồng thì có phải xác thực bổ sung không?", "Giao dịch chuyển khoản hơn 500 triệu có yêu cầu xác thực thêm không?"],
  ["Chuyển khoản dưới 500 triệu có cần xác thực thêm không?", "Rút tiền mặt trên 500 triệu có cần xác thực thêm không?"]),
 ("Đơn hàng nội thành được giao trong bao lâu?", ["Giao hàng nội thành mất bao lâu?", "Đơn nội thành bao lâu thì nhận được?"],
  ["Đơn hàng ngoại thành được giao trong bao lâu?", "Đơn hàng quốc tế được giao trong bao lâu?"]),
]
LAC = ["Thời tiết hôm nay thế nào?", "Công thức nấu phở bò?", "Ai là tác giả Truyện Kiều?",
       "Tỉ giá đô la hôm nay là bao nhiêu?", "Cách học tiếng Anh hiệu quả?"]
tok = AutoTokenizer.from_pretrained(M); mdl = AutoModel.from_pretrained(M).eval()
@torch.no_grad()
def ma(xs):
    b = tok(["query: " + x for x in xs], padding=True, truncation=True, return_tensors="pt")
    h = mdl(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
    return F.normalize((h * m).sum(1) / m.sum(1), dim=-1)
goc = ma([g for g, _, _ in BO])
noi_lai, gan = [], []
for i, (_, nl, gn) in enumerate(BO):
    for x in nl:
        s = (ma([x]) @ goc.T)[0]; noi_lai.append(dict(cau=x, cos_goc=round(s[i].item(), 4), trung_dung_goc=int(s.argmax()) == i))
    for x in gn:
        s = (ma([x]) @ goc.T)[0]; gan.append(dict(cau=x, cos_goc=round(s[i].item(), 4), goc_gan_nhat=int(s.argmax()) == i))
lac = [round((ma([x]) @ goc.T).max().item(), 4) for x in LAC]
quet = []
for k in range(80, 100):
    t = k / 100
    quet.append(dict(nguong=t,
                     trung_dung=round(sum(r["cos_goc"] >= t and r["trung_dung_goc"] for r in noi_lai) / len(noi_lai), 2),
                     tra_nham=round(sum(r["cos_goc"] >= t and r["goc_gan_nhat"] for r in gan) / len(gan), 2),
                     lac_de_trung=round(sum(c >= t for c in lac) / len(lac), 2)))
c_nl = [r["cos_goc"] for r in noi_lai]; c_g = [r["cos_goc"] for r in gan]
res = dict(model=M, noi_lai=noi_lai, gan_giong=gan, lac_de_cos_max=lac, quet_nguong=quet,
           tom_tat=dict(noi_lai_min=min(c_nl), noi_lai_max=max(c_nl), gan_giong_min=min(c_g), gan_giong_max=max(c_g),
                        so_cau_gan_giong_cao_hon_cach_noi_lai_thap_nhat=sum(c > min(c_nl) for c in c_g),
                        so_cau_gan_giong_ma_goc_gan_nhat_van_la_cau_goc=sum(r["goc_gan_nhat"] for r in gan)))
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res["tom_tat"], ensure_ascii=False))
for q in quet:
    if q["nguong"] in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96): print(q)
