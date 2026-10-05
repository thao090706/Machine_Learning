import numpy as np

# Dữ liệu
w = np.array([1, 2, -10])
x = np.array([3, 4, 1])
y = -1  # nhãn thực tế

# 1. Tính w^T x
wx = np.dot(w, x)
print("w^T x =", wx)

# 2. Xác định nhãn dự đoán bằng hàm sign
y_pred = np.sign(wx)
print("Nhãn dự đoán y_hat =", y_pred)

# 3. Kiểm tra phân lớp sai
if y_pred != y:
    print(f"Điểm dữ liệu BỊ phân lớp SAI (y_thuc_te = {y}, y_du_doan = {y_pred})")
else:
    print(f"Điểm dữ liệu được phân lớp ĐÚNG (y_thuc_te = {y}, y_du_doan = {y_pred})")

# 4. Nếu sai thì cập nhật trọng số theo quy tắc Perceptron
eta = 1
if y_pred != y:
    w_new = w + eta * y * x
    print("Trọng số cập nhật w_new =", w_new)