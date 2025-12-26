# GPU in Action

`GPU in Action`은 GPGPU(General-purpose computing on graphics processing units)를 배우고자 하는 이들에게 다양한 예제를 통해 학습 자료를 제공합니다. 이 레포지토리는 CUDA와 OpenCL을 사용한 병렬 컴퓨팅 기법을 소개하고, 이를 실제로 구현하는 방법을 제공합니다.

---

## 목차

| 제목 | 설명 |
|------|------|
| [1. vector-add](examples/1.vector-add/README.md) | 벡터 덧셈 예제 |
| [2. reduce-sum](examples/2.reduce-sum/README.md) | 합계 축소 연산 예제 |

---

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
     cmake -S . -B ./build -DUSE_OpenCL=On
     ```

3. 컴파일:
   ```bash
   cd build && make
   ```

---

이 레포지토리는 지속적으로 업데이트될 예정입니다.
