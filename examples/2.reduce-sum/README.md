# 배열의 합 예제 

이 예제는 GPU를 이용하여 배열의 합을 구하는 예시를 제공합니다.  
또한 간단한 스레드 그룹에서 사용하는 지역 변수(local, shared memory)를 활용하여 latency 를 줄이는 간단한 사용 예를 볼 수 있습니다. 

# 알고리즘 

아래는 `CUDA/kernel.cu`의 `reduce_sum` 커널이 **한 번 호출될 때**(1-pass) 각 블록에서 부분합(partial sum)을 만드는 흐름입니다.

### 1) Global memory에서 2개씩 로드 → thread별 임시 합

각 블록은 `blockDim.x`개의 스레드를 가지며, 각 스레드는 입력에서 2개 원소를 읽어 더합니다.

```
입력(input):  [ x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 ... ]

블록 0 (blockIdx=0), blockDim=8 예시

tid:          0   1   2   3   4   5    6    7
gid:          0   1   2   3   4   5    6    7
추가 로드:   +8  +8  +8  +8  +8  +8   +8   +8

thread sum:  x0+x8  x1+x9  x2+x10  x3+x11  x4+x12  x5+x13  x6+x14  x7+x15
```

### 2) Shared memory에 저장 후, stride를 줄이며 reduction

thread별 `sum`을 shared memory(`sdata[tid]`)에 저장한 뒤, stride를 절반씩 줄이면서 합칩니다.

```
sdata 초기 (각 tid가 2개씩 더해서 저장)
idx:    0       1       2        3        4        5        6        7
			 [s0]    [s1]    [s2]     [s3]     [s4]     [s5]     [s6]     [s7]
			 (x0+x8) (x1+x9) (x2+x10) (x3+x11) (x4+x12) (x5+x13) (x6+x14) (x7+x15)

stride=4: tid<4만 수행
	sdata[0]+=sdata[4]   sdata[1]+=sdata[5]   sdata[2]+=sdata[6]   sdata[3]+=sdata[7]

stride=2: tid<2만 수행
	sdata[0]+=sdata[2]   sdata[1]+=sdata[3]

stride=1: tid<1만 수행
	sdata[0]+=sdata[1]

=> 최종: sdata[0] = (이 블록이 담당한 구간의 합)
```

### 3) 블록의 부분합을 output[blockIdx]에 기록

```
if (tid == 0)
	output[blockIdx] = sdata[0]

output: [ block0_sum, block1_sum, block2_sum, ... ]
```

이후 host 코드에서는 `output`을 다음 pass의 `input`으로 스왑하고(`std::swap(d_in, d_out)`),
`current_n`이 1이 될 때까지 반복 호출하여 최종 합을 얻습니다.


# 구현 예제
| CUDA | OpenCL |
|------|--------|
| [CUDA 코드 보기](./CUDA/main.cu) | [OpenCL 코드 보기](./OpenCL/main.cpp) |
---

