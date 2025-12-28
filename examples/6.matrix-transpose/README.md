
# 행렬 전치(Transpose) 예제

이 예제에서는 2차원 행렬의 전치(transpose)를 GPU에서 수행하는 다양한 커널을 비교합니다.

## 문제 정의

입력 행렬 $A$ (M×N)가 주어졌을 때, 전치 행렬 $A^T$ (N×M)는 다음과 같이 정의됩니다.

$$
A^T[i, j] = A[j, i]
$$

## 커널 종류

1. **Naive Transpose Kernel**: 각 스레드가 단순히 $A[i, j]$를 $A^T[j, i]$로 복사합니다.
2. **Tiled Matrix Transpose (Bank Conflict)**: 공유 메모리(tile, local memory)를 사용하지만, bank conflict가 발생할 수 있는 단순 버전입니다.
3. **Tiled Matrix Transpose (No Bank Conflict)**: padding 등 기법을 활용해 bank conflict를 방지한 최적화 버전입니다.


## 메모리 접근 최적화 및 커널별 성능 이슈

### 1. 글로벌 메모리 접근 병합(Coalesced Access)

글로벌 메모리(global memory)는 bank conflict 개념이 없지만, 여러 스레드가 연속된 주소(예: 한 워프/워크그룹 내에서 0,1,2,... 순서)로 접근할 때 메모리 접근이 병합(coalesced)되어 한 번에 읽거나 쓸 수 있습니다. 이 경우 메모리 대역폭을 최대한 활용할 수 있어 성능이 높아집니다.

반대로, 스레드들이 불규칙하게(global memory에서) 멀리 떨어진 주소에 접근하면 여러 번의 메모리 트랜잭션이 발생해 성능이 저하됩니다.

#### 예시 (Coalesced vs. Non-coalesced)

```
// Coalesced (연속 접근)
스레드 0: A[0]
스레드 1: A[1]
스레드 2: A[2]
...

// Non-coalesced (불규칙 접근)
스레드 0: A[0]
스레드 1: A[100]
스레드 2: A[200]
...
```

행렬 전치와 같이 메모리 패턴이 바뀌는 연산에서는, 글로벌 메모리 접근 병합과 공유 메모리 bank conflict를 모두 고려해야 최적의 성능을 얻을 수 있습니다.

### 2. Naive 커널의 한계

Naive 커널은 각 스레드가 $A[i, j]$를 $A^T[j, i]$로 직접 복사합니다. 이때, 읽기/쓰기 모두에서 글로벌 메모리 접근이 비연속적이 되어 coalesced access가 깨지고, 메모리 대역폭을 제대로 활용하지 못해 성능이 크게 저하됩니다.

이를 개선하기 위해 local memory(공유 메모리)를 사용하는 tiled 방식이 도입됩니다.

### 3. Tiled Matrix Transpose (Bank Conflict)

공유 메모리(tile, local memory)를 사용하면 글로벌 메모리 접근은 coalesced하게 만들 수 있지만, 단순하게 구현하면 여러 스레드가 공유 메모리의 같은 bank에 몰려 bank conflict가 발생할 수 있습니다.


#### Bank란? (구체적 예시)

GPU의 공유 메모리(shared memory, OpenCL의 local memory)는 여러 개의 "bank"라는 독립적인 메모리 블록으로 구성되어 있습니다. 각 bank는 동시에 한 번에 한 주소만 접근할 수 있지만, 서로 다른 bank는 병렬로 접근이 가능합니다.

예를 들어, `float arr[16];`이 있고, 8-bank 구조라면 메모리 배치는 다음과 같습니다:

```
Bank 0: arr[0],  arr[8]
Bank 1: arr[1],  arr[9]
Bank 2: arr[2], arr[10]
Bank 3: arr[3], arr[11]
Bank 4: arr[4], arr[12]
Bank 5: arr[5], arr[13]
Bank 6: arr[6], arr[14]
Bank 7: arr[7], arr[15]
```

즉, arr[i]는 (i % 8)번 bank에 저장됩니다. 8개 스레드가 arr[0]~arr[7]에 동시에 접근하면 모두 다른 bank에 접근하므로 병렬로 처리됩니다. 하지만 arr[0], arr[8], arr[16] 등 같은 bank에 여러 스레드가 동시에 접근하면 bank conflict가 발생합니다.


##### 언제 발생하는가? (ASCII 예시, 스레드 ID 명시)

예를 들어, 8×8 tile을 공유 메모리에 저장할 때, 아래와 같이 각 스레드가 연속적으로 행(row) 방향으로 접근하면 bank conflict가 없습니다.

```
// 각 스레드가 자신의 local thread id에 해당하는 행을 읽음
tid:   0      1      2      3      4      5      6      7
	[0,0] [1,0] [2,0] [3,0] [4,0] [5,0] [6,0] [7,0]
	[0,1] [1,1] ...
	...
	[0,7] ... [7,7]
```

하지만 전치 과정에서 열(column) 방향으로 접근하면, 여러 스레드가 같은 bank에 몰려 bank conflict가 발생할 수 있습니다.

```
// 전치 시, 여러 스레드가 같은 bank(열)로 접근
tid:   0      1      2      3      4      5      6      7
	[0,0] [0,1] [0,2] [0,3] [0,4] [0,5] [0,6] [0,7]   // 모두 bank 0
	[1,0] [1,1] ... [1,7]                             // 모두 bank 1
	...
```

### 4. Tiled Matrix Transpose (No Bank Conflict) 및 해결법

bank conflict를 방지하기 위해 공유 메모리 tile에 padding을 추가하거나, 접근 패턴을 조정하는 등의 기법을 적용합니다. 이를 통해 모든 스레드가 서로 다른 bank에 접근하도록 하여 직렬화 없이 최대 대역폭을 활용할 수 있습니다.

## 구현 예제

| CUDA | OpenCL |
|--------------|----------------------|
| [CUDA 코드](./CUDA/kernel.cu) | [OpenCL 코드](./OpenCL/kernel.cl) |