"""
CÂY QUYẾT ĐỊNH ID3 - BÀI TOÁN RỦI RO TÍN DỤNG
Chỉ dùng Python thuần, không cần cài thêm thư viện.

Ý tưởng ID3 (rất ngắn gọn):
  1. Tính "độ hỗn loạn" (Entropy) của dữ liệu.
  2. Thử chia dữ liệu theo từng cột -> cột nào làm dữ liệu "sạch" nhất
     (Information Gain lớn nhất) thì chọn làm nút.
  3. Lặp lại cho từng nhánh cho đến khi nhánh chỉ còn 1 loại kết quả.
"""
import math
from collections import Counter

# ---------------------------------------------------------------
# BƯỚC 1: DỮ LIỆU (chép từ bảng trong ảnh)
# Mỗi dòng: (Tuổi, Hôn nhân, Sở hữu BĐS, Thu nhập, Rủi ro)
# ---------------------------------------------------------------
du_lieu_goc = [
    (25, "Độc thân",     "Ở cùng bố mẹ", 7000000,  0),
    (40, "Đã kết hôn",   "Nhà sở hữu",   18000000, 0),
    (35, "Từng ly hôn",  "Nhà thuê",     12000000, 1),
    (27, "Đã kết hôn",   "Ở cùng bố mẹ", 9000000,  1),
    (31, "Độc thân",     "Nhà thuê",     6000000,  1),
    (36, "Đã kết hôn",   "Nhà sở hữu",   8000000,  1),
    (48, "Độc thân",     "Nhà thuê",     7000000,  0),
    (26, "Đã kết hôn",   "Nhà thuê",     8000000,  1),
    (33, "Từng ly hôn",  "Ở cùng bố mẹ", 5000000,  1),
    (29, "Độc thân",     "Nhà thuê",     10000000, 0),
    (38, "Đã kết hôn",   "Nhà sở hữu",   15000000, 0),
    (44, "Độc thân",     "Nhà sở hữu",   14000000, 1),
    (42, "Độc thân",     "Nhà sở hữu",   10000000, 0),
    (28, "Độc thân",     "Nhà thuê",     7000000,  1),
    (30, "Đã kết hôn",   "Ở cùng bố mẹ", 6000000,  1),
]

# ---------------------------------------------------------------
# BƯỚC 2: ĐỔI SỐ THÀNH NHÓM
# ID3 chỉ làm việc với dữ liệu dạng nhóm (chữ), nên Tuổi và
# Thu nhập (là số) phải được chia thành các khoảng.
# Bạn có thể tự đổi mốc chia ở đây.
# ---------------------------------------------------------------
def nhom_tuoi(tuoi):
    if tuoi < 30:
        return "Trẻ (<30)"
    elif tuoi < 40:
        return "Trung niên (30-39)"
    return "Lớn tuổi (>=40)"

def nhom_thu_nhap(thu_nhap):
    if thu_nhap < 8000000:
        return "Thấp (<8tr)"
    elif thu_nhap < 12000000:
        return "Trung bình (8-12tr)"
    return "Cao (>=12tr)"

TEN_COT = ["Tuổi", "Hôn nhân", "Sở hữu BĐS", "Thu nhập"]

# Mỗi dòng dữ liệu là 1 dict, ví dụ:
# {"Tuổi": "Trẻ (<30)", "Hôn nhân": "Độc thân", ..., "Rủi ro": 0}
du_lieu = []
for tuoi, hn, bds, tn, rr in du_lieu_goc:
    du_lieu.append({
        "Tuổi": nhom_tuoi(tuoi),
        "Hôn nhân": hn,
        "Sở hữu BĐS": bds,
        "Thu nhập": nhom_thu_nhap(tn),
        "Rủi ro": rr,
    })

# ---------------------------------------------------------------
# BƯỚC 3: CÁC HÀM TÍNH TOÁN
# ---------------------------------------------------------------
def entropy(rows):
    """Độ hỗn loạn: =0 nếu tất cả cùng loại, =1 nếu chia đều 50/50."""
    tong = len(rows)
    dem = Counter(r["Rủi ro"] for r in rows)
    e = 0
    for so_luong in dem.values():
        p = so_luong / tong
        e -= p * math.log2(p)
    return e

def information_gain(rows, cot):
    """Chia theo cột 'cot' thì độ hỗn loạn giảm được bao nhiêu?"""
    e_truoc = entropy(rows)
    e_sau = 0
    for gia_tri in set(r[cot] for r in rows):
        nhanh = [r for r in rows if r[cot] == gia_tri]
        e_sau += len(nhanh) / len(rows) * entropy(nhanh)
    return e_truoc - e_sau

# ---------------------------------------------------------------
# BƯỚC 4: XÂY CÂY (hàm tự gọi lại chính nó - đệ quy)
# ---------------------------------------------------------------
def xay_cay(rows, cac_cot):
    ket_qua = [r["Rủi ro"] for r in rows]

    # Dừng 1: tất cả cùng 1 kết quả -> trả về kết quả đó (lá)
    if len(set(ket_qua)) == 1:
        return ket_qua[0]

    # Dừng 2: hết cột để chia -> lấy kết quả xuất hiện nhiều nhất
    if not cac_cot:
        return Counter(ket_qua).most_common(1)[0][0]

    # Chọn cột có Information Gain cao nhất
    cot_tot_nhat = max(cac_cot, key=lambda c: information_gain(rows, c))
    print(f"  -> Chọn cột '{cot_tot_nhat}' "
          f"(Gain = {information_gain(rows, cot_tot_nhat):.3f}), "
          f"số dòng: {len(rows)}")

    cay = {cot_tot_nhat: {}}
    cot_con_lai = [c for c in cac_cot if c != cot_tot_nhat]
    for gia_tri in set(r[cot_tot_nhat] for r in rows):
        nhanh = [r for r in rows if r[cot_tot_nhat] == gia_tri]
        cay[cot_tot_nhat][gia_tri] = xay_cay(nhanh, cot_con_lai)
    return cay

# ---------------------------------------------------------------
# BƯỚC 5: IN CÂY RA MÀN HÌNH CHO DỄ ĐỌC
# ---------------------------------------------------------------
def in_cay(cay, thut_le=0):
    if not isinstance(cay, dict):
        nhan = "CÓ rủi ro (1)" if cay == 1 else "KHÔNG rủi ro (0)"
        print("    " * thut_le + f"=> {nhan}")
        return
    for cot, cac_nhanh in cay.items():
        for gia_tri, con in cac_nhanh.items():
            print("    " * thut_le + f"[{cot} = {gia_tri}]")
            in_cay(con, thut_le + 1)

# ---------------------------------------------------------------
# BƯỚC 6: DỰ ĐOÁN CHO KHÁCH HÀNG MỚI
# ---------------------------------------------------------------
def du_doan(cay, khach):
    while isinstance(cay, dict):
        cot = list(cay.keys())[0]
        gia_tri = khach[cot]
        if gia_tri not in cay[cot]:
            return None  # trường hợp chưa từng thấy trong dữ liệu
        cay = cay[cot][gia_tri]
    return cay

# ---------------------------------------------------------------
# CHẠY CHƯƠNG TRÌNH
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Entropy ban đầu của toàn bộ dữ liệu:", round(entropy(du_lieu), 3))
    print("\nInformation Gain của từng cột (ở nút gốc):")
    for cot in TEN_COT:
        print(f"  {cot:12s}: {information_gain(du_lieu, cot):.3f}")

    print("\nQuá trình xây cây:")
    cay = xay_cay(du_lieu, TEN_COT)

    print("\nCÂY QUYẾT ĐỊNH:")
    in_cay(cay)

    # Thử dự đoán 1 khách hàng mới
    khach_moi = {
        "Tuổi": nhom_tuoi(27),
        "Hôn nhân": "Độc thân",
        "Sở hữu BĐS": "Nhà thuê",
        "Thu nhập": nhom_thu_nhap(7500000),
    }
    kq = du_doan(cay, khach_moi)
    print("\nKhách hàng mới:", khach_moi)
    print("Dự đoán:", {1: "CÓ rủi ro tín dụng", 0: "KHÔNG rủi ro", None: "Không xác định"}[kq])