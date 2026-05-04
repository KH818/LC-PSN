# LC-PSN: One-Bit DOA Spectrum Estimation

이 저장소는 unknown-K one-bit DOA(Direction-of-Arrival) 추정을 위한 spectrum 기반
딥러닝 실험 코드입니다.

초기 목표는 **LC-PSN: Likelihood-Consistent Probabilistic Spectrum Network**였습니다.
핵심 구상은 one-bit arcsine covariance, neural covariance refinement, Bartlett prior
spectrum, probabilistic spectrum head를 결합하고, 여기에 one-bit sign-pair likelihood
기반 BSC loss와 test-time covariance adaptation을 추가하는 것이었습니다.

다만 현재 실험 결과를 기준으로 보면, BSC와 R-TTA는 main contribution으로 주장하기에
충분히 강하지 않습니다. 따라서 이 저장소는 현재 **one-bit DOA spectrum estimation
연구용 prototype 및 baseline 코드**로 보는 것이 가장 정확합니다.

## 문제 설정

입력은 complex one-bit quantized array observation입니다.

```text
z_m(t) = sign(Re x_m(t)) + j sign(Im x_m(t))
```

모델은 이를 real-augmented snapshot tensor로 받습니다.

```text
x = [
  sign(Re z_1), ..., sign(Re z_M),
  sign(Im z_1), ..., sign(Im z_M)
]
```

입력 shape은 다음과 같습니다.

```text
[batch, 2M, snapshots]
```

현재 기본 설정은 다음과 같습니다.

```text
M = 8
K in {2, 3, 4, 5}
angle grid = 360 points over [-pi/2, pi/2]
```

모델은 spatial spectrum과 source count를 예측합니다. 최종 DOA는 예측 spectrum에서
일정 각도 이상 떨어진 peak를 선택하여 decode합니다.

## 전체 아키텍처

현재 LC-PSN 계열 모델의 전체 pipeline은 다음과 같습니다.

```text
one-bit snapshots x
-> complex one-bit tensor z
-> arcsine covariance R0
-> neural covariance refinement R_ref
-> Bartlett prior spectrum S_base
-> probabilistic spectrum heads
-> mu(theta), sigma2(theta), K_logits
```

### 1. Arcsine Covariance Front-End

one-bit snapshot은 먼저 complex sign tensor로 변환됩니다.

```python
z = sign(real) + 1j * sign(imag)
```

이후 quantized signal의 sample covariance를 correlation matrix로 정규화하고,
one-bit arcsine relation을 적용합니다.

```text
Rz = z z^H / T
Cz = normalized(Rz)
R0 = sin((pi / 2) Re(Cz)) + j sin((pi / 2) Im(Cz))
```

`R0`는 Hermitian projection을 거친 뒤 neural model로 들어갑니다.

### 2. Covariance Tokenizer와 Transformer Encoder

complex covariance matrix는 row-wise token으로 바뀝니다.

```text
R in C^{M x M}
-> concat(Re(R), Im(R)) in R^{M x 2M}
-> LayerNorm + Linear projection
-> M covariance tokens
```

Transformer encoder는 covariance token을 처리합니다. Encoder output은 두 가지로
사용됩니다.

```text
row features -> covariance residual Delta
mean-pooled feature -> spectrum and K heads
```

Covariance refiner는 Hermitian residual을 예측합니다.

```text
R_ref = Hermitian(R0 + alpha * Delta)
```

### 3. Bartlett Prior Spectrum

Refined covariance `R_ref`와 steering dictionary `A`를 사용하여 normalized Bartlett
spectrum을 계산합니다.

```text
S_base(theta) = a(theta)^H R_ref a(theta)
```

`S_base`는 neural spectrum head에 들어가는 physics-informed prior 역할을 합니다.

### 4. Probabilistic Spectrum Head와 K Head

모델 output은 다음 dictionary 형태입니다.

```python
{
    "mu": mu,
    "sigma2": sigma2,
    "mu_logits": mu_logits,
    "log_var": log_var,
    "K_logits": K_logits,
    "R_ref": R_ref,
    "S_base": S_base,
    "R0": R0,
}
```

`mu(theta)`는 최종 DOA spectrum입니다. `sigma2(theta)`는 원래 spectrum uncertainty를
표현하기 위해 추가했지만, 현재 실험에서는 sample 간 변화가 거의 없어 calibrated
uncertainty로 해석하기 어렵습니다.

K head는 pooled covariance feature, Bartlett prior, 그리고 `R_ref`의 eigenvalue feature를
사용합니다.

```text
K_logits = K_head([pooled, S_base, log eig(R_ref), log eig gaps])
```

## 초기 BSC 아이디어

초기 LC-PSN의 핵심 novelty 후보는 real-augmented Bernoulli self-consistency(BSC)
loss였습니다. 이 loss는 one-bit sign-pair likelihood를 사용하여 refined covariance가
실제 binary snapshot을 설명하도록 강제하려는 목적이었습니다.

Real-augmented sign vector `s(t)`에 대해 다음 event를 정의합니다.

```text
y_pq(t) = 1 if s_p(t) s_q(t) = +1 else 0
```

Complex covariance `R_ref`가 주어졌을 때 real-augmented covariance는 다음과 같습니다.

```text
Sigma = 1/2 * [
  Re(R_ref)  -Im(R_ref)
  Im(R_ref)   Re(R_ref)
]
```

`Sigma`를 correlation matrix `C`로 정규화하면 arcsine sign relation에 의해 다음 확률을
얻습니다.

```text
P(s_p and s_q have the same sign)
= 0.5 * (1 + (2 / pi) asin(C_pq))
```

BSC loss는 위 확률과 실제 same-sign event 사이의 binary cross entropy입니다.

```text
L_BSC = BCE(p_pq, y_pq)
```

구현은 `criterion.py`의 `real_augmented_bsc_loss`에 들어 있습니다.

## 왜 현재 BSC를 Main Contribution으로 보기 어려운가

BSC는 refined covariance가 one-bit sign pair를 설명하도록 만드는 합리적인
signal-processing 동기를 갖고 있습니다. 하지만 현재 실험에서는 BSC를 켠 모델과 끈
모델 사이에 충분히 큰 차이가 나오지 않았습니다.

현재 관찰된 내용은 다음과 같습니다.

- Standard DOA metric이 BSC 유무에 따라 크게 달라지지 않습니다.
- K accuracy가 BSC에 의해 일관되게 개선되지 않습니다.
- Held-out `L_BSC(R_ref, x)`가 BSC-trained checkpoint와 no-BSC checkpoint 사이에서 거의
  같습니다.
- 일부 SNR에서는 BSC가 조금 낫지만, 다른 SNR에서는 거의 같거나 약간 나쁩니다.

가장 가능성 높은 해석은 arcsine covariance front-end가 이미 sign-pair 통계의 대부분을
담고 있다는 것입니다. One-bit snapshot이 `R0`로 압축된 뒤에는 BSC가 추가로 개선할 수
있는 여지가 작습니다.

따라서 현재 BSC는 검증된 main contribution이라기보다, 실험적으로 탐색한 auxiliary
regularizer로 보는 것이 안전합니다.

## 초기 R-TTA 아이디어

두 번째 초기 아이디어는 test-time covariance adaptation, 즉 R-TTA였습니다.

```text
1. forward x -> R_ref, mu, sigma2, K_logits
2. update only R_ref using grad L_BSC(R_ref, x)
3. Hermitian-project R_ref
4. recompute mu, sigma2, K_logits with forward_from_cov()
```

이때 model weight는 업데이트하지 않고, covariance output인 `R_ref`만 test-time에
조정합니다.

## 왜 현재 R-TTA를 사용하지 않는가

현재 R-TTA는 downstream DOA metric과 K metric을 악화시킵니다. 관찰된 현상은 다음과
같습니다.

- Covariance update 동안 `L_BSC`가 거의 감소하지 않습니다.
- `R_ref`의 작은 변화가 `forward_from_cov`에서 큰 spectrum/K 변화로 증폭될 수 있습니다.
- Spectrum head와 K head는 model-generated covariance feature 위에서 학습되었고,
  test-time optimization으로 만들어진 off-manifold covariance에 robust하게 학습되지
  않았습니다.

즉 BSC objective는 거의 움직이지 않는데, downstream predictor는 불안정해질 수 있습니다.
이는 유용한 adaptation이라기보다 off-manifold robustness 문제에 가깝습니다.

R-TTA는 다음 조건을 만족하기 전까지 improvement로 보고하지 않는 것이 맞습니다.

- held-out BSC loss가 의미 있게 감소해야 합니다.
- 모든 SNR에서 DOA MAE가 악화되지 않아야 합니다.
- 모든 SNR에서 K accuracy가 악화되지 않아야 합니다.

## 현재 실용 Baseline

현재 가장 안정적인 baseline은 BSC나 R-TTA에 의존하지 않는 spectrum-based
LC-PSN/TransMUSIC-style 모델입니다.

```text
one-bit snapshots
-> arcsine covariance
-> neural covariance refinement
-> Bartlett prior spectrum
-> spectrum head + K head
-> separated peak decoding
```

이 모델은 앞으로의 one-bit DOA 연구에서 실험 base로 사용할 가치가 있습니다. 이유는
다음과 같습니다.

- one-bit arcsine covariance correction을 사용합니다.
- spectrum 기반 DOA inference를 수행합니다.
- unknown-K classification을 포함합니다.
- eigenvalue-aware K estimation을 사용합니다.
- few-snapshot 및 low-SNR 평가가 가능합니다.

## 주요 파일

```text
model.py                       TransMusic, LCPSN 모델 정의
criterion.py                   Spectrum loss, BSC loss, DOA matching utility
dataset.py                     Synthetic one-bit DOA dataset 생성 및 로딩
physics.py                     ULA steering vector, signal generation, MUSIC baseline
train.py                       Training script
eval.py                        Evaluation script 및 held-out BSC 출력
diagnose_lcpsn_spectrum.py     Spectrum behavior 진단 CSV 생성
AGENTS.md                      내부 프로젝트 메모와 현재 연구 상태
```

## Evaluation 실행

Checkpoint 평가:

```bash
python eval.py --ckpt best_lcpsn_1bit.pt
```

Low-SNR 평가:

```bash
python eval.py --ckpt best_lcpsn_1bit.pt --snrs -10 -5 0 5 10
```

`eval.py`는 다음 metric을 출력합니다.

```text
K Accuracy
DOA MAE
DOA RMSE
Success Rate @ 5 deg
K-correct DOA MAE/RMSE
K confusion matrix
Eval BSC L(R_ref, x)
```

## Few-Snapshot 평가

`eval.py`는 새 test file을 만들 때 `SNAPSHOTS` 값을 사용합니다. Snapshot 수는 생성되는
파일명에 포함됩니다.

```text
test_snr_0_T20.h5
test_snr_5_T20.h5
test_snr_10_T20.h5
```

이 방식은 서로 다른 snapshot 수로 생성된 test file을 실수로 재사용하는 문제를 막기 위한
것입니다.


## 현재 상태

이 저장소는 완성된 benchmark package가 아니라, spectrum 기반 one-bit DOA estimation
아이디어를 개발하고 검증하기 위한 연구 코드입니다.

현재 상태를 정직하게 요약하면 다음과 같습니다.

- Arcsine covariance + spectrum model은 유용한 baseline입니다.
- BSC는 real-augmented sign-pair 형태로 구현되었지만, 현재 empirical gain은 main novelty
  claim을 뒷받침하기에 너무 작습니다.
- R-TTA는 현재 regression이며 improvement로 보고하면 안 됩니다.
- 새로운 novelty는 현재 BSC ablation보다 더 강한 probabilistic 또는 signal-processing
  evidence를 제공해야 합니다.
