# 누적합(Navie)

이 문서에서는 GPGPU를 이용하여 누적합을 구현하는 가장 기초적인 방식을 제공합니다. 

## 알고리즘 

누적합(prefix sum, scan)은 입력 $A$에 대해

- Inclusive scan: $S[i] = \sum_{k=0}^{i} A[k]$
- Exclusive scan: $S[i] = \sum_{k=0}^{i-1} A[k]$ (보통 $S[0]=0$)

을 구하는 연산입니다. CPU에서는 단순 루프(이전 값 누적)로 쉽게 구현되지만, GPU에서는 여러 스레드가 동시에 실행되므로 “이전 원소의 결과”를 순차적으로 참조하는 형태를 그대로 옮길 수 없습니다.

이 예제는 **가장 단순한 2-pass 방식**으로 누적합을 구현합니다.

1) **워크그룹 내부 scan(부분 누적합)**
- 각 work-item이 연속 원소를 묶어서 처리합니다. 여기서는 `int4` 한 개를 읽어 4개 원소를 한 스레드가 담당합니다.
- 먼저 스레드 내부에서 `int4`의 4개 성분에 대해 순차 누적을 만들어 둡니다.
	- `data.y += data.x; data.z += data.y; data.w += data.z;`
	- 이 시점에서 `data.w`는 “이 work-item이 담당한 4개 원소의 합”입니다.
- 다음으로 work-group 전체에서, 각 work-item의 합(`data.w`)에 대해 scan을 수행해 **work-item 단위 prefix**를 얻습니다.
	- 아래 `scan1()`은 local memory를 사용한 Hillis–Steele 형태의 **inclusive scan**입니다.
- inclusive 결과 `val`에서 자기 구간의 합(`data.w`)을 빼면, 자기 work-item 이전까지의 **exclusive 오프셋**이 됩니다.
	- `offset = val - data.w`
	- 최종적으로 `dst[id] = data + offset`을 하면, work-group 내부에 대해서는 올바른 누적합이 됩니다.

2) **그룹 합(groupsum) 전파를 위한 준비**
- 각 work-group의 마지막 work-item이 `val`(=해당 그룹의 총합)을 `groupsum[group+1]`에 저장합니다.
- `groupsum[0]=0`을 넣어두면, 이후 `groupsum` 자체를 prefix-scan 했을 때 각 그룹의 시작 오프셋으로 바로 쓸 수 있습니다.

3) **워크그룹 간 오프셋 적용(uniform update)**
- `groupsum` 배열에 대해 prefix-scan을 한 뒤(입력 크기에 따라 CPU에서 해도 되고, 그룹 수가 크면 재귀적으로 GPU에서 한 번 더 scan),
	`uniformUpdate()`에서 `gid`(그룹 id)에 해당하는 오프셋 `groupsum[gid]`를 그룹 내 모든 원소에 더해 최종 결과를 만듭니다.

아래는 3)의 전개를 아주 단순화한 그림입니다.

```text
입력은 work-group 단위로 나뉨 (각 그룹은 여러 work-item이 int4 등을 처리)

	Group 0            Group 1            Group 2
 [a0 a1 a2 a3]     [b0 b1 b2 b3]     [c0 c1 c2 c3]   ...
		  |                 |                 |
		  | scan4: 그룹 내부 scan (부분 결과)  |
		  v                 v                 v
  dst 부분결과0      dst 부분결과1      dst 부분결과2
  (그룹 내부만      (그룹 내부만      (그룹 내부만
	올바른 누적)      올바른 누적)      올바른 누적)

  그리고 각 그룹의 총합을 groupsum에 기록

  groupsum (raw):  [0, sum(G0), sum(G1), sum(G2), ...]
							^   ^        ^        ^
							|   |        |        |
					  groupsum[0]  groupsum[1] groupsum[2] ...

  groupsum에 대해 prefix-scan 수행(=그룹 시작 오프셋이 됨)

  groupsum (scanned):
	 [0,
	  sum(G0),
	  sum(G0)+sum(G1),
	  sum(G0)+sum(G1)+sum(G2), ...]

  uniformUpdate: gid != 0 인 그룹의 dst에 groupsum[gid]를 더해 전파

	dst_final(Group 1) = dst_partial(Group 1) + groupsum[1]
	dst_final(Group 2) = dst_partial(Group 2) + groupsum[2]
	...
```

## 구현 예제
| CUDA | OpenCL |
|------|--------|
|[CUDA 예제](./CUDA/main.cu) | [OpenCL 예제](./OpenCL/main.cpp) |
---
