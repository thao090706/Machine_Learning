import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])
y = 1
eta = 1

# 1. Kiểm tra phân lớp
wx = np.dot(w, x)
y_pred = np.sign(wx)
print("w^T x =", wx)
print("y_hat =", y_pred)

if y_pred != y:
    print(f"Phân lớp SAI (y_thuc_te={y}, y_du_doan={y_pred})")
    # 2. Cập nhật trọng số
    w_new = w + eta * y * x
    print("w_new =", w_new)
    # 3. Tính lại w^T x
    wx_new = np.dot(w_new, x)
    print("w_new^T x =", wx_new)
    print("y_hat_new =", np.sign(wx_new))
else:
    print(f"Phân lớp ĐÚNG (y_thuc_te={y}, y_du_doan={y_pred})")