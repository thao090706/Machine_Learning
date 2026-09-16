import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. ĐỌC DATASET
# ============================================

df = pd.read_csv("kimmich.csv")

print("Dataset:")
print(df.head())
print("\nSố dòng:", len(df))
print("\nThông tin:")
print(df.info())

# ============================================
# 2. X VÀ y
# ============================================

# X = 3 chỉ số chuyền bóng
X = df[["Cmp", "Att", "Cmp%"]]

# y = Key Pass
y = df["KP"]

# ============================================
# 3. CHIA TRAIN / TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================
# 4. MÔ HÌNH CƠ BẢN - LINEAR REGRESSION
# ============================================

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

linear_model.fit(X_train, y_train)

pred_train = linear_model.predict(X_train)
pred_test = linear_model.predict(X_test)

print("\n========== LINEAR REGRESSION ==========")
print("Train RMSE:",
      np.sqrt(mean_squared_error(y_train, pred_train)))
print("Test RMSE :",
      np.sqrt(mean_squared_error(y_test, pred_test)))
print("Train R2:",
      r2_score(y_train, pred_train))
print("Test R2 :",
      r2_score(y_test, pred_test))

# ============================================
# 5. CỐ TÌNH TẠO OVERFITTING
# ============================================

overfit_model = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

overfit_model.fit(X_train, y_train)

pred_train = overfit_model.predict(X_train)
pred_test = overfit_model.predict(X_test)

train_rmse = np.sqrt(
    mean_squared_error(y_train, pred_train)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, pred_test)
)

print("\n========== POLYNOMIAL DEGREE 3 ==========")
print("Train RMSE:", train_rmse)
print("Test RMSE :", test_rmse)
print("Train R2:", r2_score(y_train, pred_train))
print("Test R2 :", r2_score(y_test, pred_test))

# ============================================
# 6. 5-FOLD CROSS VALIDATION
# ============================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    overfit_model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

cv_rmse = -scores

print("\n========== 5-FOLD CROSS VALIDATION ==========")

for i, score in enumerate(cv_rmse, 1):
    print(f"Fold {i}: RMSE = {score:.4f}")

print("Mean CV RMSE:", cv_rmse.mean())
print("Std CV RMSE :", cv_rmse.std())

# ============================================
# 7. THỬ DEGREE 1 -> 10
# ============================================

degrees = range(1, 6)
train_rmse = []
cv_rmse = []

for degree in degrees:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", LinearRegression())
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_train)
    train_rmse.append(
        np.sqrt(mean_squared_error(y_train, pred))
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    cv_rmse.append(-scores.mean())

    print(
        f"Degree {degree}: "
        f"Train RMSE={train_rmse[-1]:.4f}, "
        f"CV RMSE={cv_rmse[-1]:.4f}"
    )

best_degree = degrees[np.argmin(cv_rmse)]

print("\nDegree tốt nhất theo 5-Fold CV:", best_degree)

if 20 <= cv_rmse[np.argmin(cv_rmse)] <= 50:
    print("CV RMSE nằm trong khoảng mục tiêu 20-50.")
else:
    print("CV RMSE chưa nằm trong khoảng 20-50; RMSE phụ thuộc dữ liệu và không nên ép bằng cách chọn degree.")

# ============================================
# 8. BIỂU ĐỒ OVERFITTING
# ============================================

plt.figure(figsize=(10, 6))

plt.plot(
    list(degrees),
    train_rmse,
    marker="o",
    label="Train RMSE"
)

plt.plot(
    list(degrees),
    cv_rmse,
    marker="o",
    label="5-Fold CV RMSE"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title(
    "Overfitting và K-Fold Cross Validation\n"
    "Joshua Kimmich - Dự đoán Key Pass"
)

plt.xticks(list(degrees))
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================
# 9. CHỌN DEGREE THEO CV RMSE
# ============================================

best_index = int(np.argmin(cv_rmse))
best_degree = list(degrees)[best_index]

print("\n========== KẾT QUẢ ==========")
print("Degree có CV RMSE nhỏ nhất:", best_degree)
print("CV RMSE:", cv_rmse[best_index])

print(
    "\nX: Cmp, Att, Cmp%"
    "\ny: KP (Key Pass)"
)
