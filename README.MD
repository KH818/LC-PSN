# One-Bit Unknown-K DOA Estimation: TranSpectrum에서 LC-PSN / RA-PSN까지

이 저장소는 **1-bit 관측 기반 unknown-K DOA 추정**을 다룬다. 처음에는
TranSpectrum / TransMUSIC 스타일의 neural spectrum estimator에서 출발했고,
이후 LC-PSN, RA-PSN, IWBO likelihood 실험까지 확장했다.

현재까지의 핵심 결론은 다음이다.

```text
arcsine covariance + neural spectrum inference는 이미 매우 강한 baseline이다.
likelihood consistency는 arcsine covariance에 이미 들어있는 second-order
sign-pair 정보를 넘어서는 추가 신호를 제공할 때만 의미가 있다.
```

현재 paper-facing 방향은 LC-PSN/BSC가 아니라 다음에 가깝다.

```text
PSN / RA-PSN:
  finite-snapshot arcsine covariance uncertainty
  reliability-aware covariance regularization
  spectrum-based unknown-K inference
```

코드에는 호환성을 위해 `LCPSN`이라는 클래스명이 남아 있지만, 현재 가장 강한
모델은 **RA-PSN-Cov**로 보는 것이 더 적절하다.

---

## 문제 설정

입력은 complex one-bit array snapshot이다.

```text
z_m(t) = sign(Re x_m(t)) + j sign(Im x_m(t))
```

이를 real-augmented sign vector로 쓰면:

```text
s(t) = [sign(Re z_1), ..., sign(Re z_M),
        sign(Im z_1), ..., sign(Im z_M)]
```

목표는 다음과 같다.

```text
K in {2, 3, 4, 5}인 unknown source count 추정
각 source의 DOA 추정
입력은 1-bit snapshot만 사용
```

주요 평가 지표:

```text
K Accuracy
DOA MAE / RMSE
Success Rate @ 5 deg
K-correct RMSE
SNR별 / K별 breakdown
tail RMSE: p90 / p95 / p99 / max
```

---

## TranSpectrum에서 LC-PSN으로

### TranSpectrum의 기본 아이디어

초기 모델은 TransSpectrum / TransMUSIC 구조를 기반으로 했다.

```text
one-bit snapshots
-> arcsine covariance R0
-> covariance row tokenizer
-> Transformer encoder
-> angle grid 위의 spectrum 예측
-> peak decoding
```

핵심 아이디어는 hand-crafted MUSIC peak picking 대신, covariance-like
representation을 neural network가 읽고 task-adapted spectrum을 예측하게 하는
것이었다.

다만 한계가 있었다.

```text
1. K 추정이 간접적이었다.
   spectrum peak threshold나 peak count에 의존했다.

2. one-bit likelihood consistency가 명시적으로 들어가지 않았다.
   supervised spectrum은 학습했지만, 예측한 spectrum/covariance가 실제
   sign-pair 통계를 설명하는지는 직접 강제하지 않았다.
```

### LC-PSN의 아이디어

LC-PSN은 TranSpectrum 구조 위에 probabilistic spectrum과 likelihood
consistency를 추가하려는 시도였다.

의도한 pipeline:

```text
one-bit snapshots x [B, 2M, T]
-> arcsine covariance R0
-> optional covariance refinement R_ref
-> Bartlett / physics prior spectrum S_base
-> Transformer covariance features
-> neural spectrum mu(theta)
-> optional sigma2(theta)
-> K_logits
-> separated top-K peak decoding
```

처음 생각한 novelty는 다음과 같았다.

```text
1. Probabilistic spectrum output
   mu(theta), sigma2(theta)

2. Explicit K estimation
   K_logits로 K in {2, 3, 4, 5}를 직접 분류

3. Covariance refinement
   R0 -> R_ref로 covariance를 보정한 뒤 spectrum 계산

4. Physics-informed prior
   R_ref에서 얻은 Bartlett spectrum S_base를 neural spectrum head의
   prior/residual input으로 사용

5. Likelihood consistency
   BSC loss를 통해 예측 covariance가 one-bit sign-pair statistics와
   일치하도록 유도
```

---

## 아키텍처 요약

현재 `LCPSN` 코드 경로는 다음과 같다.

```text
x [B, 2M, T]
  |
  | real/imag -> complex z [B, M, T]
  v
one-bit arcsine covariance R0 [B, M, M]
  |
  | real-augmented sign uncertainty U0 [B, 2M, 2M]
  v
CovarianceTokenizer:
  [Re(R), Im(R)]
  또는 [Re(R), Im(R), log(U0)]
  |
Transformer encoder over sensor-row tokens
  |
row_out -> Delta covariance
  |
R_ref = R0 + alpha * Delta
  |
Bartlett spectrum S_base = a(theta)^H R_ref a(theta)
  |
mu_head:
  neural residual + beta * logit(S_base)
  -> mu(theta)
  |
log_var_head:
  -> sigma2(theta)
  단, 현재 sigma2는 calibration이 검증되지 않았으므로 diagnostic으로만 해석
  |
K_head:
  pooled covariance feature + S_base + eigenvalues + eigengaps
  -> K_logits
```

Decoding:

```text
K_hat = argmax(K_logits) + K_min
DOA_hat = mu(theta)에서 top K_hat separated peaks
```

---

## TranSpectrum 대비 K Accuracy가 좋아진 이유

K accuracy 향상은 주로 BSC 때문이 아니라, 아키텍처와 decoding 방식 변화
때문이다.

가장 중요한 요인:

```text
1. 명시적인 K classification head
   spectrum threshold로 K를 세는 대신 K_logits를 직접 학습한다.

2. eigenvalue / eigengap feature
   K head가 R_ref의 eigenvalue와 log-eigengap을 입력으로 받는다.
   이는 source count 구조와 직접 관련이 있다.

3. separated top-K decoding
   K_hat을 먼저 정한 뒤, mu(theta)에서 top-K separated peaks를 선택한다.
   threshold 기반 peak counting보다 안정적이다.

4. balanced multi-K simulation
   K in {2, 3, 4, 5}를 균형 있게 학습하여 supervised model-order
   classification을 수행한다.

5. physics prior spectrum
   S_base가 강한 covariance-based prior를 제공하므로 neural head는 전체
   spectrum을 처음부터 만들기보다 residual correction을 학습한다.
```

강한 LC-PSN / RA-PSN checkpoint의 T=200 K accuracy는 대략 다음 수준이다.

```text
KAcc ~= 0.95 - 0.98
```

따라서 이후 연구의 핵심은 K accuracy를 더 올리는 것보다 **RMSE와 tail
failure를 줄이는 것**으로 이동했다.

---

## BSC가 실패한 이유

BSC는 one-bit sign-pair statistics에 대한 Bernoulli self-consistency loss다.

의도:

```text
예측 covariance에서 arcsine law로 sign-pair probability를 계산하고,
empirical sign-pair statistics와의 불일치를 줄인다.
```

실험 결론:

```text
BSC는 no-BSC neural spectrum baseline 대비 의미 있는 개선을 주지 못했다.
standard, low-SNR, few-snapshot setting에서 모두 headline contribution으로
보기 어렵다.
```

가능한 원인:

```text
1. arcsine covariance가 이미 second-order sign-pair 정보를 대부분 포함한다.

2. one-bit snapshot을 R0로 압축한 뒤에는 BSC가 추가로 활용할 정보가 적다.

3. BSC는 covariance consistency를 맞추지만, 최종 metric은 DOA peak
   localization과 K estimation이다.

4. 작은 covariance 변화가 downstream spectrum head에서 예측 불안정으로
   증폭될 수 있다.
```

결론:

```text
BSC는 diagnostic 또는 negative ablation으로만 남긴다.
main novelty로 사용하지 않는다.
```

---

## R-TTA가 실패한 이유

R-TTA는 test-time에 `R_ref`만 BSC loss로 업데이트하고, 다시 spectrum과 K를
계산하는 방식이다.

실험 결론:

```text
R-TTA는 downstream DOA와 K metric을 악화시켰다.
```

가능한 원인:

```text
1. BSC를 줄이는 것이 DOA error를 줄인다는 보장이 없다.

2. eval-time covariance update가 R_ref를 training manifold 밖으로 보낸다.

3. forward_from_cov / spectrum decoding은 작은 covariance perturbation에도
   민감하다.

4. 모델이 test-time covariance optimization에 robust하도록 학습되지 않았다.
```

결론:

```text
R-TTA는 headline result로 사용하지 않는다.
negative ablation으로만 언급한다.
```

---

## RA-PSN: Finite-Snapshot Arcsine Uncertainty

현재 가장 강한 방향은 BSC가 아니라 **finite-snapshot reliability of arcsine
covariance**다.

real-augmented sign pair에 대해:

```text
v_pq(t) = s_p(t) s_q(t)
tau_hat_pq = mean_t v_pq(t)
C_hat_pq = sin((pi / 2) tau_hat_pq)
```

delta method로 uncertainty를 추정한다.

```text
Var(C_hat_pq)
~= ((pi / 2) cos((pi / 2) tau_hat_pq))^2
    * (1 - tau_hat_pq^2) / T
```

정의:

```text
U0_pq = Var(C_hat_pq)
```

U0 사용 방식은 두 가지를 테스트했다.

```text
RA-PSN-B2:
  log(U0)를 covariance tokenizer feature로 concatenate

RA-PSN-Cov:
  uncertainty-weighted covariance consistency 사용
  L_uncov = |R_ref - R0|^2 / (U0 + eps)
```

실험 결론:

```text
U0 feature concatenation만으로는 실패했다.
uncertainty-weighted covariance consistency는 localization 개선 신호를 보였다.
```

---

## 주요 LC-PSN / RA-PSN 실험

### 4-way neural spectrum 비교

조건:

```text
T = 200
M = 8
SNR in {0, 5, 10}
K in {2, 3, 4, 5}
```

결과:

```text
Method                  KAcc    RMSE    SR@5   KcorrRMSE   p90    p95
PSN baseline            0.9766  2.309   0.891   2.187      5.224  8.214
RA-PSN-B2               0.9727  2.719   0.839   2.622      6.826 10.382
RA-PSN-Cov 0.001        0.9779  2.079   0.902   1.921      4.848  7.068
RA-PSN-Cov 0.005        0.9727  2.139   0.900   1.961      4.998  7.585
```

핵심 결론:

```text
RA-PSN-Cov 0.001이 현재 가장 강한 neural variant다.
RA-PSN-B2는 negative/simple ablation으로 본다.
```

### K=5 tail 개선

K=5는 가장 중요한 stress case다.

```text
K=5 RMSE:
  PSN baseline       4.153 deg
  RA-PSN-B2          4.340 deg
  RA-PSN-Cov 0.001   3.461 deg
  RA-PSN-Cov 0.005   3.832 deg

K=5 p90:
  PSN baseline       10.267 deg
  RA-PSN-Cov 0.001    7.339 deg
```

이는 uncertainty-weighted covariance consistency가 high-K / tail failure를
줄이는 데 의미가 있음을 보여준다.

---

## Paired Train-Seed Diagnostics

### Seed 2023, Eval Seed 9000

```text
baseline:
  KAcc=0.9714, MAE=1.5134, RMSE=2.1252, KcorrRMSE=2.0212

uncov0001:
  KAcc=0.9714, MAE=1.4481, RMSE=2.0806, KcorrRMSE=1.8519

delta:
  KAcc=+0.0000
  RMSE=-0.0446 deg
  KcorrRMSE=-0.1694 deg
```

seed 2023에서는 전체 RMSE 개선은 작지만, K-correct RMSE 개선은 뚜렷하다.
다만 p90/p95/p99는 일관되게 좋아지지는 않았다.

### Seed 2024, Eval Seed 9000

```text
baseline:
  KAcc=0.9792, MAE=1.6164, RMSE=2.2946, KcorrRMSE=2.2174

uncov0001:
  KAcc=0.9635, MAE=1.4697, RMSE=2.1293, KcorrRMSE=1.9345

delta:
  KAcc=-0.0156
  RMSE=-0.1653 deg
  KcorrRMSE=-0.2829 deg
```

seed 2024에서는 localization 개선이 더 분명하지만, K accuracy가 약간
희생됐다.

### Combined Paired Seeds

```text
baseline:
  KAcc=0.9753, MAE=1.5649, RMSE=2.2099, KcorrRMSE=2.1193

uncov0001:
  KAcc=0.9674, MAE=1.4589, RMSE=2.1049, KcorrRMSE=1.8932

delta:
  KAcc=-0.0078
  MAE=-0.1060 deg
  RMSE=-0.1050 deg
  KcorrRMSE=-0.2261 deg
```

해석:

```text
uncertainty-weighted covariance consistency는 localization, 특히 K-correct
RMSE에서 의미 있는 신호가 있다.
하지만 K accuracy가 항상 좋아지는 것은 아니므로 multi-seed 검증이 필요하다.
```

---

## Mixed-T 실험

snapshot 수를 섞어서 학습하는 mixed-T 모델도 테스트했다.

```text
T in {20, 50, 100, 200}
```

결론:

```text
Mixed-T training은 calibration과 K accuracy를 악화시켰다.
특히 원래 LC-PSN/RA-PSN이 학습된 T=200 setting에서 성능이 흔들렸다.
```

진단:

```text
T가 바뀌면 finite-snapshot 통계, covariance reliability, K-head calibration이
모두 바뀐다. mixed-T 학습은 하나의 모델이 여러 regime을 동시에 맞추게 만들어
주 실험의 attribution을 흐린다.
```

현재 규칙:

```text
main model은 T=200 고정으로 학습한다.
T variation은 robustness evaluation으로만 사용한다.
```

---

## IWBO 실험

IWBO는 one-bit likelihood 기반 weighting/objective를 neural spectrum learning에
활용하려는 시도다. 이는 과거 BSC framing과는 구분한다.

### Pairwise IWBO

pairwise IWBO는 differentiable pairwise sign likelihood를 사용한다.

```text
tau_model = (2 / pi) asin(rho(C(theta)))
log p_pair = sum_{p<q} Bernoulli(tau_hat_pq | tau_model_pq)
```

장점:

```text
batch training 가능
미분 가능
계산량이 낮음
```

한계:

```text
arcsine covariance / BSC와 정보가 많이 겹친다.
```

### Full Orthant IWBO

정확한 one-bit likelihood는 Gaussian orthant probability다.

```text
log p(Y | theta, p, sigma2)
= sum_t log P(diag(y_t) w_t > 0)
```

진단용 구현은 GHK Monte Carlo를 사용했다.

Full orthant sanity 결과:

```text
Fixed full orthant:
  true_beats_5deg   = 0.9806
  true_beats_10deg  = 0.9944
  true_beats_random = 0.9972
  Kmax overfit rate = 0.0259

Profiled p/sigma2 full orthant:
  true_beats_5deg   = 0.9806
  true_beats_10deg  = 0.9917
  true_beats_random = 1.0000
  Kmax overfit rate = 0.0259

Marginalized p/sigma2 full orthant:
  true_beats_5deg   = 0.9833
  true_beats_10deg  = 0.9917
  true_beats_random = 1.0000
  Kmax overfit rate = 0.0259
```

해석:

```text
full orthant likelihood 자체에는 정보가 있다.
하지만 현재 GHK estimator는 direct training에 쓰기에는 아직 불안정하다.
```

### Full Orthant Rescore 실패

LC-PSN local rescore 결과:

```text
Pairwise fine rescore:
  LCPSN RMSE    = 2.2804 deg
  rescored RMSE = 2.2274 deg
  delta         = -0.0529 deg

Fixed full orthant rescore:
  LCPSN RMSE    = 1.9485 deg
  rescored RMSE = 2.1253 deg
  delta         = +0.1769 deg

Profile full orthant rescore:
  LCPSN RMSE    = 1.9485 deg
  rescored RMSE = 2.1812 deg
  delta         = +0.2327 deg

Marginal full orthant rescore:
  LCPSN RMSE    = 1.9485 deg
  rescored RMSE = 2.1778 deg
  delta         = +0.2294 deg
```

결론:

```text
full orthant는 true-vs-wrong sanity는 통과했지만 local rescore에서는 실패했다.
현재 상태에서는 full orthant GHK를 every-batch training loss로 쓰지 않는다.
```

### IWBO-clean 학습

IWBO-clean은 pairwise IWBO가 기존 auxiliary loss들을 대체할 수 있는지 보기
위한 실험이었다.

Loss:

```text
L = L_spec + 0.5 L_K + lambda_iwbo L_IWBO_pairwise
```

비활성화한 항목:

```text
GNLL
log_var regularization
uncov
PSD
covariance alignment
Hermitian consistency
```

10 epoch pilot:

```text
control_e10:
  KAcc=0.9309, RMSE=5.9605 deg, KcorrRMSE=5.7258 deg

no_gnll_var_e10:
  KAcc=0.9326, RMSE=6.2071 deg, KcorrRMSE=5.9356 deg

iwbo_clean_e10:
  KAcc=0.9283, RMSE=5.1198 deg, KcorrRMSE=4.7668 deg
```

pilot 해석:

```text
GNLL/log_var만 제거하면 localization이 악화된다.
pairwise IWBO는 early localization regularizer로는 효과가 있다.
```

Full train:

```text
LC-PSN baseline:
  KAcc=0.9620, MAE=1.5162 deg, RMSE=2.1328 deg, KcorrRMSE=1.9396 deg

RA-PSN-Cov / uncov0005:
  KAcc=0.9611, MAE=1.4655 deg, RMSE=2.0857 deg, KcorrRMSE=1.8786 deg

IWBO-clean:
  KAcc=0.9606, MAE=1.6119 deg, RMSE=2.2632 deg, KcorrRMSE=2.0227 deg
```

결론:

```text
IWBO-clean은 LC-PSN baseline이나 RA-PSN-Cov를 이기지 못했다.
pairwise IWBO는 clean replacement loss로 쓰면 안 된다.
```

### Full-IWBO Candidate Weighting Sanity

목적:

```text
LC-PSN 예측 주변 후보들 중 full orthant score가 낮은 RMSE 후보를 더 높게
rank하는지 확인
```

Main run, `best_lcpsn_1bit.pt`, 60 samples:

```text
pairwise:
  Spearman=0.6010, improve_rate=0.7667, delta=-0.4974 deg

orthant fixed:
  Spearman=0.3528, improve_rate=0.6333, delta=-0.1780 deg

orthant profile:
  Spearman=0.3491, improve_rate=0.6000, delta=-0.1785 deg

orthant marginal:
  Spearman=0.3357, improve_rate=0.6000, delta=-0.1757 deg
```

High-accuracy check, particles=256, snapshot_limit=32:

```text
pairwise:
  Spearman=0.5645, improve_rate=0.6667, delta=-0.3246 deg

orthant fixed:
  Spearman=0.4154, improve_rate=0.5833, delta=-0.2377 deg
```

RA-PSN-Cov / uncov0005 checkpoint:

```text
pairwise:
  Spearman=0.5517, improve_rate=0.7833, delta=-0.5151 deg

orthant fixed:
  Spearman=0.3523, improve_rate=0.6833, delta=-0.3075 deg
```

결론:

```text
full orthant는 positive ranking signal은 있지만 pairwise ranking을 이기지 못했다.
full orthant training은 estimator나 proposal design을 개선한 뒤 다시 본다.
```

### Pairwise IWBO + RA-PSN-Cov

Loss:

```text
L = L_spec + 0.5 L_K + lambda_uncov L_uncov + lambda_iwbo L_IWBO_pairwise
```

Seed 2023 diagnostics:

```text
baseline:
  KAcc=0.9714, MAE=1.5134, RMSE=2.1252, KcorrRMSE=2.0212, SR5=0.9089

uncov0001:
  KAcc=0.9714, MAE=1.4481, RMSE=2.0806, KcorrRMSE=1.8519, SR5=0.9023

IWBO+uncov:
  KAcc=0.9701, MAE=1.4866, RMSE=2.0905, KcorrRMSE=1.9599, SR5=0.8997
```

IWBO+uncov vs uncov0001:

```text
KAcc        -0.0013
MAE         +0.0386 deg
RMSE        +0.0100 deg
KcorrRMSE   +0.1080 deg
SR5         -0.0026
```

K별 결과:

```text
K=2: RMSE +0.0057 deg, 거의 동일
K=3: RMSE +0.1434 deg, 악화
K=4: RMSE +0.2828 deg, 악화
K=5: RMSE -0.3920 deg, 개선
```

해석:

```text
IWBO+uncov는 전체 평균 성능에서 uncov0001을 넘지 못했다.
하지만 K=5 tail-error를 줄이는 신호는 있다.
따라서 pairwise IWBO는 weak, annealed, 또는 high-K-focused auxiliary로
다시 테스트할 가치가 있다.
```

---

## 현재 결론

1. **LC-PSN의 K accuracy 향상은 주로 explicit K head와 eigengap feature
   때문이다.**

2. **BSC와 R-TTA는 실패한 방향이다.**
   arcsine covariance가 sign-pair 정보를 이미 많이 포함하고 있고, covariance
   consistency를 줄이는 것이 DOA peak를 개선한다는 보장이 없었다.

3. **현재 가장 강한 방향은 RA-PSN-Cov다.**
   uncertainty-weighted covariance consistency는 localization과 K-correct RMSE를
   개선하는 신호가 있다.

4. **IWBO는 main replacement loss로는 아직 부족하다.**
   pairwise IWBO는 early training과 K=5 tail에는 신호가 있지만, naive하게 쓰면
   최종 평균 RMSE를 개선하지 못한다.

5. **Full orthant IWBO는 이론적으로 매력적이지만 현재 구현은 불안정하다.**
   true-vs-wrong sanity는 통과했지만, candidate ranking에서는 pairwise보다 약했다.

---

## 추천 다음 실험

우선순위:

```text
1. RA-PSN-Cov multi-seed validation
   seeds: 2023, 2024, 2025, 2026, 2027

2. RA-PSN-Cov lambda sweep
   lambda_uncov in {0.0, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01}

3. Dense / high-K stress test
   K in {4, 5}
   min_sep_deg in {4, 6, 8}

4. Pairwise IWBO as weak auxiliary
   lambda_iwbo in {0.001, 0.003, 0.005}
   annealing 또는 K=5-focused diagnostic 고려

5. Full orthant estimator 개선
   full IWBO training은 candidate ranking이 pairwise를 이긴 뒤 다시 진행
```

---

## 주요 파일

```text
model.py
  LCPSN / TransMusic architecture

train.py
  baseline 및 RA-PSN-Cov 학습

train_iwbo_clean.py
  pairwise IWBO-clean / IWBO+uncov 학습 variant

eval.py
  fixed-SNR evaluation

diagnose_lcpsn_spectrum.py
  spectrum diagnostics 및 per-sample CSV 생성

diagnose_iwbo_likelihood.py
  pairwise / full orthant likelihood sanity

diagnose_full_iwbo_candidate_weighting.py
  full-IWBO candidate ranking sanity

uncertainty.py
  finite-snapshot arcsine covariance uncertainty utilities
```

---

## Naming Note

코드에는 여전히 `LCPSN` 클래스명이 남아 있다. 논문 작성에서는 다음처럼 구분하는
것이 좋다.

```text
PSN:
  baseline probabilistic spectrum network

RA-PSN:
  reliability-aware probabilistic spectrum network

RA-PSN-Cov:
  RA-PSN with uncertainty-weighted covariance consistency

LC-PSN:
  likelihood-consistency version의 역사적 이름
```

LC-PSN/BSC라는 이름은 개발 과정이나 negative ablation을 설명할 때만 사용하는
것이 안전하다.
