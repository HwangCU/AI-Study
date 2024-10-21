# Weight initialization

> 제프리 힌톤 왈

우리는 멍청한 방식으로 가중치를 초기화 하고 있었다

가중치를 잘 초기화 해야하는데 고정하고 사용하고 있었다.

![alt text](image.png)

가중치를 초기화하는 방법이 성능에 중요한 영향을 끼친다.

### Weight를 초기화 하는 방법
- 0이 아닌 값으로 한다.
- Restricted Boltzmann Machine(RBM)

> RBM이란?
![alt text](image-1.png)
서로같은 층에 있는 노드들은 연결이 되어 있지 않고 서로 다른 층과 연결되어 있는 구조를 말한다.또한 순방향 x > y 구조와 y > x' 을 하는 역방향 구조를 지닌다. 이는 인코딩 디코딩 구조와 비슷하다.

### 어떻게 RBM을 Weight Initialization 했을 까?
pre-training Step을 했다.
![alt text](image-2.png)

2개의 레이어를 RBM으로 학습을 시킨다. 층을 여러개 쌓을 경우 그 전의 weight는 고정 시킨다.

- 하지만 RBM은 요즘은 잘 사용하지 않는다. 복잡하고 다른 간단한 Weight Initialization이 있기 때문이다.

### Xavier/He Initialization

> Xavier
간단히 수식을 통해서 Weight를 초기화 하는 방법

- Xavier Normal Initialization
![alt text](image-3.png)
레이어의 input수와 ouput수를 통해 초기값을 구할 수 있다.

- Xavier Uniform Initialization
![alt text](image-4.png)

>He Initialization

![alt text](image-5.png)