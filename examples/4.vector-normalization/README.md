# 벡터 정규화 예제

이 예제에서는 GPGPU를 활용하여 Vector Normalization 을 수행하는 예제를 보여줍니다.

이 예제는 [2. 배열의 합](../2.reduce-sum/README.md)에서 다룬 **reduction(리덕션)** 개념을 그대로 활용합니다.
정규화는 “벡터의 크기(노름)를 구한 뒤, 각 성분을 그 크기로 나누기”이기 때문에,
GPU에서는 보통 다음 두 단계로 구현합니다.

1) 벡터의 노름(norm) 계산: 전체 성분에 대한 합(리덕션)이 필요
2) 각 성분 스케일링: element-wise 연산

## 알고리즘 

### 1) 수식 정의

벡터 $v \in \mathbb{R}^N$의 $L2$ 노름은

$$\|v\|_2 = \sqrt{\sum_{i=0}^{N-1} v[i]^2}$$

정규화(normalization) 결과 $\hat{v}$는

$$\hat{v}[i] = \frac{v[i]}{\|v\|_2}$$

입니다.

> 예외 케이스: $\|v\|_2 = 0$ 이면 나눗셈이 불가능하므로,
> 보통은 모든 성분을 0으로 두거나(가장 단순), 입력을 그대로 유지하는 정책을 선택합니다.

### 2) GPU 구현(개요)

GPU에서 핵심 병목은 $\sum v[i]^2$처럼 **전체 합**을 구해야 한다는 점입니다.
즉, 아래처럼 “element-wise”와 “reduction”을 결합한 파이프라인이 됩니다.

```text
입력 벡터 v
 v: [v0 v1 v2 v3 v4 v5 ... vN-1]

1단계) sumsq = Σ(v[i]^2)
	- (커널 A) 각 스레드가 v[i]^2를 계산
	- (커널 B) reduce-sum을 여러 pass로 반복하여 최종 합 1개를 얻음

2단계) norm = sqrt(sumsq)
	- (보통) host가 norm 값을 읽거나, device에 1개 값을 둔 채 다음 커널에서 사용

3단계) v_hat[i] = v[i] / norm
	- (커널 C) element-wise 나눗셈
```

여기서 1단계는 [2. 배열의 합](../2.reduce-sum/README.md)와 동일한 형태의 리덕션을 사용합니다.

### 3) 리덕션 패스(ASCII 그림)

local size를 $L$이라고 하면, 한 work-group(또는 CUDA block)은 보통 $2L$개를 읽어 부분합을 만들고,
work-group 내부에서 local/shared memory로 트리 리덕션을 수행합니다.

```text
예: L=8인 경우(개념도)

각 스레드가 2개씩 로드해 임시합을 만든 뒤 local/shared에 저장
 tmp[lid] = v[gid]^2 + v[gid+L]^2

tmp 초기:
 idx:   0    1    2    3    4    5    6    7
			 [t0] [t1] [t2] [t3] [t4] [t5] [t6] [t7]

stride=4:
 tmp[0]+=tmp[4], tmp[1]+=tmp[5], tmp[2]+=tmp[6], tmp[3]+=tmp[7]

stride=2:
 tmp[0]+=tmp[2], tmp[1]+=tmp[3]

stride=1:
 tmp[0]+=tmp[1]

=> 블록/work-group 결과 1개 생성
 partial[group_id] = tmp[0]
```

이렇게 만들어진 `partial[]`은 길이가 work-group 개수만큼이며,
`partial[]`에 대해 동일한 reduce-sum을 반복 호출해 길이가 1이 될 때까지 줄이면
최종 `sumsq`가 됩니다.

### 4) 정규화 단계(스케일링)

`norm = sqrt(sumsq)`를 얻으면, 마지막은 간단히 element-wise로 나눕니다.

```text
norm: scalar 1개

v:     [v0    v1    v2    ...]
v_hat: [v0/n  v1/n  v2/n  ...]

각 i에 대해 스레드 1개가 담당
```

# 구현 예제
| CUDA | OpenCL |
|------|--------|
|[CUDA 예제](./CUDA/main.cu) | [OpenCL 예제](./OpenCL/main.cpp) |
---
