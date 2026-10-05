import numpy as np
from itertools import product
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


# ---------------- Lớp Perceptron tự cài đặt ----------------
class Perceptron:
    def __init__(self, eta=1.0, max_epochs=100, shuffle=True, seed=0):
        self.eta, self.max_epochs = eta, max_epochs
        self.shuffle, self.seed = shuffle, seed
        self.w, self.errors_ = None, []

    @staticmethod
    def _add_bias(X):
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def fit(self, X, y):
        X = self._add_bias(np.asarray(X, float))
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(X.shape[1])
        best_w, best_err = self.w.copy(), np.inf   # "pocket": giữ w tốt nhất
        self.errors_ = []
        for _ in range(self.max_epochs):
            idx = rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            for i in idx:
                if y[i] * np.dot(self.w, X[i]) <= 0:
                    self.w += self.eta * y[i] * X[i]
            err = np.sum(np.where(X @ self.w >= 0, 1, -1) != y)
            self.errors_.append(err)
            if err < best_err:
                best_err, best_w = err, self.w.copy()
            if err == 0:
                break
        self.w = best_w
        return self

    def predict(self, X):
        X = self._add_bias(np.asarray(X, float))
        return np.where(X @ self.w >= 0, 1, -1)


def metrics(y_true, y_pred):
    return dict(acc=accuracy_score(y_true, y_pred),
                prec=precision_score(y_true, y_pred),
                rec=recall_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred))


# ---------------- Dữ liệu ----------------
data = load_breast_cancer()
X, y = data.data, np.where(data.target == 0, 1, -1)  # +1 = ác tính (malignant), -1 = lành tính
print("Kích thước:", X.shape, "| Ác tính:", (y == 1).sum(), "| Lành tính:", (y == -1).sum())

# Chia train / validation / test = 60/20/20 (giữ tỉ lệ lớp)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

# Chuẩn hóa (fit trên train)
scaler = StandardScaler().fit(X_train)
Xtr, Xva, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

# ---------------- Baseline: không chuẩn hóa ----------------
base = Perceptron(eta=1.0, max_epochs=100, shuffle=False).fit(X_train, y_train)
print("\n[Baseline - không chuẩn hóa] trên test:", {k: round(v, 4) for k, v in metrics(y_test, base.predict(X_test)).items()})

# ---------------- Tinh chỉnh siêu tham số trên tập validation ----------------
best = None
for eta, epochs, shuf in product([0.001, 0.01, 0.1, 1.0], [50, 100, 300, 1000], [False, True]):
    m = Perceptron(eta, epochs, shuf).fit(Xtr, y_train)
    f1 = metrics(y_val, m.predict(Xva))["f1"]
    acc = metrics(y_val, m.predict(Xva))["acc"]
    if best is None or (f1, acc) > best[0]:
        best = ((f1, acc), dict(eta=eta, max_epochs=epochs, shuffle=shuf))
print("\nSiêu tham số tốt nhất (theo F1 trên validation):", best[1], "| F1 val =", round(best[0][0], 4))

# ---------------- Huấn luyện lại trên train+val, đánh giá trên test ----------------
Xfull = np.vstack((Xtr, Xva)); yfull = np.concatenate((y_train, y_val))
final = Perceptron(**best[1]).fit(Xfull, yfull)
y_pred = final.predict(Xte)
res = metrics(y_test, y_pred)
print("\n[Mô hình cuối - đã chuẩn hóa + tinh chỉnh] trên test:")
for k, name in [("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1-score")]:
    print(f"  {name:<10}: {res[k]:.4f}")
print("Ma trận nhầm lẫn [[TN FP],[FN TP]]:\n", confusion_matrix(y_test, y_pred, labels=[-1, 1]))