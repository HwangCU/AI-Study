# Linear Regression

### Hypothesis
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>H</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>W</mi>
  <mi>x</mi>
  <mo>+</mo>
  <mi>b</mi>
</math>

### cost function
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>c</mi>
  <mi>o</mi>
  <mi>s</mi>
  <mi>t</mi>
  <mo stretchy="false">(</mo>
  <mi>W</mi>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>m</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>m</mi>
  </munderover>
  <msup>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">(</mo>
      <mi>H</mi>
      <mo stretchy="false">(</mo>
      <msup>
        <mi>x</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo stretchy="false">)</mo>
      <mo>&#x2212;</mo>
      <msup>
        <mi>y</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo data-mjx-texclass="CLOSE">)</mo>
    </mrow>
    <mn>2</mn>
  </msup>
</math>

- H(x): 주어진 x값에 대해 예측을 어떻게 할 것인가
- cost: H(x)가 y를 얼마나 잘 예측했는가
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# For reproducibility
torch.manual_seed(1)
<torch._C.Generator at 0x7fa29837ef90>
Data
We will use fake data for this example.

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])


W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#hypothesis
hypothesis = x_train * W + b
#Cost
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost) # tensor(4.6667, grad_fn=<MeanBackward1>)

# Gradient Descent
optimizer = optim.SGD([W, b], lr=0.01)
optimizer.zero_grad()
cost.backward()
optimizer.step()

hypothesis = x_train * W + b
cost = torch.mean((hypothesis - y_train) ** 2)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)
# optimizer = torch.optim.SGD([W, b], lr=0.01)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad() # Gradient 초기화
    cost.backward() # gradient 계산
    optimizer.step() # step으로 개선

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
#Hypothesis
hypothesis = model(x_train)
# Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
cost.backward()
optimizer.step()

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
model = LinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    # 100번마다 로그 출력
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))