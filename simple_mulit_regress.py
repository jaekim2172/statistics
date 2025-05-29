import numpy as np
import matplotlib.pyplot as plt

# 데이터
X1 = np.array([2, 4, 6, 8, 10])
X2 = np.array([9, 8, 7, 6, 5])
Y = np.array([50, 60, 65, 80, 95])

# 다중회귀 계수
b0 = 20
b1 = 4
b2 = 3

# 예측값 계산 (다중회귀)
Y_hat = b0 + b1 * X1 + b2 * X2

# Y 평균
Y_mean = np.mean(Y)

# 잔차 제곱합 (다중회귀)
SS_res = np.sum((Y - Y_hat) ** 2)
# 총 제곱합
SS_tot = np.sum((Y - Y_mean) ** 2)
# 결정계수 다중회귀
R_squared = 1 - SS_res / SS_tot

# ----- X1과 Y의 단순 회귀 R^2 계산 -----
b1_only = np.cov(X1, Y)[0,1] / np.var(X1)  # 단순 선형회귀 기울기
b0_only = np.mean(Y) - b1_only * np.mean(X1)
Y_hat_X1 = b0_only + b1_only * X1
SS_res_X1 = np.sum((Y - Y_hat_X1) ** 2)
R_squared_X1 = 1 - SS_res_X1 / SS_tot

# ----- X2와 Y의 단순 회귀 R^2 계산 -----
b2_only = np.cov(X2, Y)[0,1] / np.var(X2)  # 단순 선형회귀 기울기
b0_only = np.mean(Y) - b2_only * np.mean(X2)
Y_hat_X2 = b0_only + b2_only * X2
SS_res_X2 = np.sum((Y - Y_hat_X2) ** 2)
R_squared_X2 = 1 - SS_res_X2 / SS_tot

# 결과 출력
print(f"다중회귀 R^2 = {R_squared:.4f}")
print(f"X1과 Y 단순회귀 R^2 = {R_squared_X1:.4f}")
print(f"X2과 Y 단순회귀 R^2 = {R_squared_X2:.4f}")

# 시각화
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(X1, Y, color='blue', label='Actual Y')
plt.plot(X1, Y_hat_X1, color='red', label='Predicted Y by X1')
plt.title(f'X1 vs Y\nR^2 = {R_squared_X1:.3f}')
plt.xlabel('X1')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.scatter(X2, Y, color='blue', label='Actual Y')
plt.plot(X2, Y_hat_X2, color='red', label='Predicted Y by X2')
plt.title(f'X2 vs Y\nR^2 = {R_squared_X2:.3f}')
plt.xlabel('X2')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
plt.scatter(range(len(Y)), Y, color='blue', label='Actual Y')
plt.plot(range(len(Y)), Y_hat, color='red', marker='o', linestyle='-', label='Predicted Y (Multiple Regression)')
plt.axhline(Y_mean, color='green', linestyle='--', label='Mean of Y')
plt.title(f'Multiple Regression\nR^2 = {R_squared:.3f}')
plt.xlabel('Index')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
