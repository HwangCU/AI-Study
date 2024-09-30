# Docker Instruction
컨테이너 기반의 오픈소스 가상화 플랫폼
## Docker란
Container-Based Virtualization System
Virtualization은 현실에 있지 않은 환경을 만들어 준다.
- 다양한 프로그램, 실행환경을 컨테이너로 추상화 하고 동일한 인터페이스를 제공하여 프로그램의 배포 및 관리를 단순하게 해준다.
- 백엔드 프로그램, 데이터베이스 서버, 메시지 큐등 어떤 프로그램도 Container를 추상화 할 수 있고, PC, AWS, Google Cloud등 어디에서든 실행가능

### Container
- 이전 방식 VMware 등은 사용법이 간단 하지만, 무겁고 느리다.
- CPU의 가상화 기술(HVM)를 이용한 반 가상화 방식 Xen을 사용
- 전체 OS를 가상화 하지 않고 Host OS를 가상화
- 하나의 서버에 여러 개의 Container 가능
- **이미지를 실행한 상태를 의미**
### Image
- Container에 필요한 파일과 설정 값 등을 포함
- Image 안에 여러 개의 Container
- Docker hub을 통해 필요한 Image을 다운 가능
- **도커에 필요한 서버 프로그램, 소스코드, 실행 파일로 프로그램을 구동하기 위한 종합선물세트**

![alt text](image-3.png)

## Docker 설치 가이드
- Docker 설치 전

먼저 컴퓨터 하드웨어가 가상화를 지원해야만 가능합니다. 그래서 작업 관리자(ctrl + alt + del)에 진입 후 성능에 들어가서 가상화를 확인할 수 있습니다.
![alt text](image-1.png)
![alt text](image-2.png)

- Docker 설치

[Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

위 링크를 통해 다운로드
or
```
winget install -e --id Docker.DockerDesktop
```

- 설치 확인
Docker desktop 들어가서 정상 작동 확인
cmd창에 아래 명령어로 확인
```
docker -v
```
## Docker 사용법
- Image 다운로드

![alt text](image-4.png)
혹은 cmd에
```
docker pull deeplearningzerotoall/pytorch
```
이미지를 받으셨으면 `docker images` 명령어를 쳐서 도커 이미지의 존재를 확인
```
docker images
REPOSITORY				TAG	...
hello-world				latest	... 
deeplearningzerotoall/pytorch		latest	...
```
### 도커 이미지로부터 컨테이너 실행하고 끄기
도커 이미지로부터 컨테이너를 생성하고 실행합니다. 이때 --name 뒤에 원하는 이름을 설정할 수 있습니다. 이번 가이드에서는 pt 로 하겠습니다.
```
docker run -i -t --name pt -p 8888:8888 -p 8097:8097 deeplearningzerotoall/pytorch /bin/bash
```
도커 컨테이너를 만들었고 root 계정으로 컨테이너 내부 쉘에 접속한 상태입니다.

```
root@[고유번호]:~#
```

### 도커 컨테이너 자유롭게 드나들기
주의: docker run 명령어는 한번 만 실행하시면 됩니다. 종료된 컨테이너를 다시 실행하려면 아래와 같이 실행합니다.
```
docker start pt
```
이 단계는 컨테이너를 키기만 했을 뿐입니다. 컨테이너 터미널에 접속하려면 아래와 같이 실행합니다. 그러면 컨테이너 내부 터미널로 이동하게 됩니다.

```
docker attach pt
root@[고유번호]:~#
```

### 가상환경에 Git Clone 하기
![alt text](image-5.png)

```
git clone https://github.com/deeplearningzerotoall/PyTorch.git
```
```
root@[고유번호]:~# git clone https://github.com/deeplearningzerotoall/PyTorch.git
```
완료가 되면 이제 PyTorch 디렉토리로 이동합니다.
```
root@[고유번호]:~# cd PyTorch
root@[고유번호]:~/PyTorch# 
```
필수 패키지를 설치해줍니다.
```
root@[고유번호]:~/PyTorch# pip install --upgrade pip
root@[고유번호]:~/PyTorch# pip install -r requirements.txt
```
jupyter notebook을 실행합니다.
```
jupyter notebook --ip 0.0.0.0 --allow-root
```
jupyter notebook 처음 실행후 아래 이미지 처럼 도커 터미널에 긴 토큰이 생기는데 해당 토큰을 복사해둡니다
![alt text](image-6.png)

`127.0.0.1:8888` url을 통해 접속 후 복사한 토큰을 비밀번호로 입력.

![alt text](image-7.png)

go!!! PJT
## VS Code에서 Docker사용하기
1. **Vscode Extension 설치**
![alt text](image-8.png)

- vscode를 켜고 좌측에 테트리스 처럼 생긴 아이콘을 클릭해서 'remote'라고 검색 후 'Remote Development' 설치
- 해당 팩에는 WSL, Dev Containers, Remote-SSH, Remote-Tunnels 4가지가 포함되어 있음

2. **Docker 컨테이너 실행**
![alt text](image-9.png)
- 컨테이너에 vscode python extension이 설치되어 있지 않기 때문에 설치하라는 안내 문구가 뜨면 설치하면 됨

## VS Code에서 오류가 나올시
> python 설치가 안됐다고 나오는 경우

Ubuntu 20.04에 Python 3용 pip를 설치
```
sudo apt update
sudo apt install python3-pip
```
> import Lib가 안되는 경우

![alt text](image-10.png)

**모듈을 가져오는 경로(path)에 해당하는 모듈이 존재하지 않기 때문**

그렇다면 내가 가져오려는 모듈은 어디에 있는 걸까?
![alt text](image-11.png)

해당 경로 복사

Docker CLI에 `echo $PATH` 명령어로 PATH확인
`sudo vi ~/.bashrc` 명령어 입력 후** 맨 하단에 PATH 추가**
```python
# PATH=$PATH:추가할 경로
PATH=$PATH:/opt/conda/lib/python3.6/site-packages
```
:wq! 입력하여 저장 후 종료

다시 `echo $PATH` 명령어 입력하면 추가한 경로를 확인할 수 있다.