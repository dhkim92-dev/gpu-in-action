# GPU in Action

`GPU in Action`은 GPGPU(General-purpose computing on graphics processing units)를 배우고자 하는 이들에게 다양한 예제를 통해 학습 자료를 제공합니다. 이 레포지토리는 CUDA와 OpenCL을 사용한 병렬 컴퓨팅 기법을 소개하고, 이를 실제로 구현하는 방법을 제공합니다.

## 목차

| 제목 | 설명 |
|------|------|
| [1. vector-add](examples/1.vector-add/README.md) | 벡터 덧셈 예제 |
| [2. reduce-sum](examples/2.reduce-sum/README.md) | 합계 축소 연산 예제 |
| [3. dot-product](examples/3.dot-product/README.md) | 벡터 내적 예제 |
| [4. vector-normalization](examples/4.vector-normalization/) | 벡터 정규화 예제 | 
| [5. prefix-sum-naive](examples/5.prefix-sum-naive/) | 누적합 예제 |
| [6. matrix-transpose](examples/6.matrix-transpose//) | 행렬 전치 예제 |

## 빌드 방법

이 프로젝트는 CMake를 사용합니다. 다음 단계를 따라 빌드할 수 있습니다:

1. 프로젝트 클론:
   ```bash
   git clone <repository_url>
   cd gpu-in-action
   ```

2. 빌드 디렉토리 생성 및 CMake 실행:
   - CUDA 지원 활성화:
     ```bash
     cmake -S . -B ./build -DUSE_CUDA=On
     ```
   - OpenCL 지원 활성화:
     ```bash
     cmake -S . -B ./build -DUSE_OPENCL=On
     ```

3. 컴파일:
   ```bash
   cd build && make
   ```

빌드 결과 build/examples/number.lectur_name 의 형태로 디렉터리가 생성되고, 하위에 lecture_name_CUDA, lecture_name_OpenCL    
실행 바이너리가 생성됩니다

## 이 자료에 대하여
기본적인 GPGPU에 필요한 내용을 다루기는 하지만 언어의 사용법 등을 상세히 설명하는 자료는 아닙니다.  
CUDA 또는 OpenCL 등의 고급 사용법은 다른 자료를 찾아주세요.  
이 레포지토리는 지속적으로 업데이트될 예정입니다.
