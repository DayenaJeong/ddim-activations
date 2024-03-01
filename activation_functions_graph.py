import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def prelu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def softplus(x):
    return np.log(1 + np.exp(x))

# 데이터 준비
x = np.linspace(-10, 10, 1000)

# 활성화 함수들을 그래프에 그리기
plt.figure(figsize=(12, 8))
plt.plot(x, sigmoid(x), label='sigmoid')
# Softmax 제외
plt.plot(x, tanh(x), label='tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, elu(x), label='ELU')
plt.plot(x, prelu(x), label='PReLU')
plt.plot(x, softplus(x), label='Softplus')
plt.plot(x, swish(x), label='Swish')
plt.plot(x, mish(x), label='Mish')
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend(loc='lower right', fontsize=12) # 범례 위치 및 크기 조절
plt.grid(True)
plt.ylim(-1, 2)
plt.xlim(-4, 4)
plt.show()
