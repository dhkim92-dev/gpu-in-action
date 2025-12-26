# 벡터 내적

이 예제는 GPU를 이용하여 벡터의 내적을 구하는 예제를 보여줍니다.
이전 예제의 응용으로 [1. 두 벡터의 합](../1.vector-add/README.md) 과 [2. 배열의 합](../2.reduce-sum/README.md) 를 참고하세요.

# 알고리즘 

두 벡터 $A$, $B$의 내적(dot product)은 아래와 같이 정의됩니다.

$$\mathrm{dot}(A,B)=\sum_{i=0}^{N-1} A[i]\cdot B[i]$$

OpenCL 버전에서는 이 합을 한 번에 한 스레드가 모두 더하는 대신,
GPU의 많은 work-item이 **부분합(partial sum)**을 만든 뒤, **리덕션(reduction)**으로 최종 합을 구합니다.

## 1) 전체 흐름(개요)

ASCII로 보면 이런 파이프라인입니다.

```text
입력 벡터
 A: [a0 a1 a2 a3 a4 a5 a6 a7 ... aN-1]
 B: [b0 b1 b2 b3 b4 b5 b6 b7 ... bN-1]

1-pass (커널: 곱 + 워크그룹 내 리덕션)
 partial: [p0 p1 p2 ... p(G-1)]      (G = work-group 개수)

2-pass 이상 (커널: reduce_sum 반복)
 partial: [q0 q1 ...] -> ... -> [final]

결과
 dot(A,B) = final[0]
```

여기서 핵심은:

- **첫 번째 커널**은 $A[i]\cdot B[i]$를 만든 뒤, 그 합을 work-group 단위로 1개 값으로 줄입니다.
- 그 결과(부분합 배열)는 길이가 $N$보다 훨씬 작아지고, 이를 **reduce_sum 커널로 반복 리덕션**해서 1개 값으로 만듭니다.

## 2) 첫 번째 패스: 곱 + 워크그룹 내 리덕션

이 예제의 첫 커널(`dot_product_reduce_first`)은 성능을 위해 work-item 하나가 보통 **2개 원소**를 처리합니다.
로컬 크기(local size)를 $L$이라고 하면, 한 work-group은 총 $2L$개의 원소 구간을 담당합니다.

```text
work-group 하나가 담당하는 구간(예: L=4)

global index:  0   1   2   3   4   5   6   7
	       |---첫 4개---| |---다음 4개---|
work-item lid:  0   1   2   3   0   1   2   3

각 work-item(lid=k)가 하는 일:
  local_sums[k] = A[gid] * B[gid] + A[gid+L] * B[gid+L]
		 (범위를 넘으면 해당 항은 생략)
```

그 다음 단계는 **로컬 메모리(local memory)**에 쌓인 `local_sums[]`를 반씩 줄여가며 더하는 트리 리덕션입니다.

```text
local_sums (L=8 예시)

초기:   [s0 s1 s2 s3 s4 s5 s6 s7]
stride4: [s0+s4  s1+s5  s2+s6  s3+s7   x   x   x   x]
stride2: [t0+t2  t1+t3   x      x      x   x   x   x]
stride1: [u0+u1   x      x      x      x   x   x   x]

마지막에 lid==0만 output[group_id]에 기록
```

즉, **work-group마다 결과 1개**가 나오고, 이 값들이 모인 배열이 “부분합 배열”이 됩니다.

## 3) 두 번째 패스 이후: reduce_sum 반복

첫 패스가 끝나면 배열 길이는 $N$에서 $G$로 줄어듭니다.
하지만 아직 1개 값이 아니므로, 같은 방식의 리덕션 커널(`reduce_sum`)을 반복 실행합니다.

```text
N=1024, L=256인 경우(개념도)

pass1: 1024개 -> 2L=512개씩 그룹화 -> G=2개 partial 생성
	[p0 p1]

pass2: 2개 -> 1개
	[final]
```

중요한 포인트:

- 각 패스마다 입력/출력 버퍼를 **핑퐁(ping-pong)** 하며 스왑해서 덮어쓰지 않습니다.
- `barrier(CLK_LOCAL_MEM_FENCE)`로 work-group 내부 동기화를 해줘야 리덕션이 안전합니다.



# 구현 예제
| CUDA | OpenCL |
|------|--------|
| [CUDA 코드 보기](./CUDA/main.cu) | [OpenCL 코드 보기](./OpenCL/main.cpp) |
---
