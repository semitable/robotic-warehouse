# 비교 알고리즘 구현 기록 — MAPPO / SePS / HetDPS

2026-10-13 Claude Code. 코드: `marl_baselines.py`. 실행: `train_from_scratch.py --modes mappo seps hetdps` 또는 `run_from_scratch.ps1 -Modes ...`.
기존 IPPO/PGCT/oracle/all/wrong 코드는 변경하지 않았다. 세 알고리즘 모두 기존 `T.train` 루프를 그대로 쓰고, 한 프로세스 안에서 agent network 생성과 `update_ippo`만 교체한다.

## 공통 조건 (IPPO/PGCT와 동일하게 유지한 것)

- **환경·관측·보상**: E1 11×10, 4 agents/2 teams, home .5, individual reward, semantic 1593차원 관측, action mask.
- **rollout·평가**: `collect_rollout`, `evaluate` 함수를 그대로 쓴다. 평가는 51,200 step마다 20×500, seed 5,000,001부터. train/eval 지표 계산도 같다.
- **예산과 seed**: 같은 env step 예산과 seed 목록을 쓴다. W&B Step 축은 모두 env steps다.
- **PPO 설정(공통 예산 기준)**: rollout 1024, sequence 64, epochs 4, minibatch 8 chunks, clip .2, value clip .2, lr 2e-4, entropy .05, γ .99, λ .95, grad norm .5, Adam eps 1e-5.
- **네트워크 용량**: IPPO와 같은 semantic CNN encoder + GRU(128) + linear head. 공식 구현의 MLP/LayerNorm 구조는 쓰지 않는다.
- **advantage 정규화**: 전체 batch(모든 agent) 단위로 한다. MAPPO·HetDPS 방식이다.
- **IPPO와 다른 점**: 기존 IPPO/PGCT trainer의 `skip_zero_reward_actor_update`는 baseline에 적용하지 않는다.
- **초기 가중치**: 같은 seed라도 구조가 달라 IPPO와 초기값이 다르다. 비교표 "Same init" 열에는 `n/a`로 표시된다.
- **checkpoint**: 모든 network는 `agents[0]` state_dict에 들어 있다. 배정 정보는 `*.baseline.pt`, `baseline_state.json`에 저장된다.

## MAPPO — Yu et al., NeurIPS 2022 (marlbenchmark/on-policy)

**구현한 핵심**
- 모든 agent가 공유하는 recurrent actor
- 공유 중앙 recurrent critic V(s)
- ValueNorm(β=0.99999, debiasing, var≥1e-2)
- normalized target에 대한 Huber(δ=10) + value clipping(두 손실 중 큰 값)
- actor/critic 별도 Adam
- PPO clip
- denormalized value로 GAE 계산
- time-limit bootstrap도 중앙 critic으로 계산

**원 구현과 다른 점**
- **중앙 상태**: 이 환경에는 전역 상태가 없다. 그래서 4개 agent의 국소 관측을 공유 CNN으로 인코딩해 이어 붙이고, 평가 대상 agent의 one-hot을 더했다. on-policy의 MPE는 관측 연결을 쓰고, SMAC의 agent-specific state는 ID를 포함한다. reward가 individual이라 agent 구분이 필요하다.
- **agent ID**: actor 입력에 agent one-hot을 추가했다(공유 정책에서 agent 구분).
- **하이퍼파라미터**: 원 논문 기본값(lr 5e-4/7e-4, epoch 5–15, mini-batch 1, chunk 10, grad norm 10, entropy .01, feature LayerNorm) 대신 공통 예산 설정을 썼다.
- **중앙 critic 값 계산 시점**: rollout 뒤 기록한 joint 관측으로 다시 계산한다. 첫 minibatch replay 오차는 약 2e-7이었다.

**정보 조건**: critic이 다른 agent의 관측을 쓰는 **CTDE, 추가 정보 허용 비교군**이다. 실행 시 행동은 자기 관측만 쓴다.

## SePS — Christianos et al., ICML 2021 (uoe-agents/seps)

**공식 기본값대로 구현한 것**
- **network 구성**: agent 수(N)만큼 network(actor + local critic). actor와 critic 모두 선택적으로 공유한다.
- **클러스터링 전**: agent 0 → network 0, 나머지는 무작위 배정한다(`sample_laac`). 클러스터링 전에도 RL 학습은 계속한다(`delay_training=False`).
- **pretraining 길이**: `pretraining_steps=5000`×8 env = 40,000 env step이며 예산에 포함된다.
- **VAE 구조**: `LinearVAE`. encoder는 agent one-hot을 받아 2×10을 출력하고, decoder는 [z, obs, act]를 받아 [next_obs, rew]를 복원한다.
- **VAE 학습**: loss = MSE(sum) + 1e-4·KLD(sum), 10 epochs, batch 128, Adam 3e-4.
- **클러스터링**: encoder 평균에 KMeans(k-means++, n_init=10)를 적용한다. K는 Davies–Bouldin 기준으로 2..N−1에서 고른다.
- **클러스터링 후**: cluster label을 network index로 쓰고, network는 재초기화하지 않는다.

**원 구현과 다른 점**
- **학습기**: A2C(n_steps 5, 8 env)가 아니라 공통 recurrent PPO를 쓴다. SePS의 기여는 학습기와 독립적인 선택적 공유다.
- **병렬 env**: env가 1개라서, 병렬 env마다 달랐던 클러스터링 전 무작위 배정을 rollout마다 다시 뽑는 방식으로 대체했다.
- **buffer**: reservoir 50,000 transition으로 제한했고 관측은 float16으로 저장한다. episode 경계 transition은 next_obs가 없어 제외한다.
- **VAE 은닉 크기**: `HIDDEN` 상수 값은 원문에서 확인하지 못해 64를 썼다.
- **클러스터링 라이브러리**: scikit-learn이 환경에 없어 KMeans·Davies–Bouldin을 같은 기본값의 numpy 코드로 구현했다. 사용자 환경에 패키지를 설치하지 않았다.

**추가 비용**: 40,000 env step의 pretraining(예산 안), VAE 학습 1회 약 9초(GPU, 212,110 parameters, 50k transitions), RAM 약 300MiB.

## HetDPS — Heterogeneity-based Dynamic Parameter Sharing, AAMAS 2026 (Harry67Hu/HetDPS)

**공식 설정대로 구현한 것**
- **network 구성**: N개 network(actor + local critic). 처음에는 모든 agent가 network 0을 쓴다.
- **buffer**: FIFO, 최근 20×5×32 = 3,200 env step × N agent.
- **측정 주기**: 3,200 env step마다(`reparameter_interval=20` update × 160 transition) 측정한다.
- **VAE**: `BiHierarchicalVAE`(구조 그대로, HIDDEN 32, z 40)를 이어서 학습한다. 매번 새 Adam(lr 5e-4), 10 epochs, batch 128, kl 1e-4, MSE(sum).
- **VAE 입력·출력**: encoder는 agent one-hot과 condition [local_state, act one-hot]을 받는다. 복원 대상은 [next_local_state, next_obs, rew_pad]다.
- **거리**: 버퍼의 모든 상태 × 5개 행동 조건에서 agent별 Gaussian 사이 Wasserstein 거리를 계산해 평균한다(`fixed_obs_all_acts`).
- **군집**: AffinityPropagation(−WD), preference는 중앙값, damping .5.
- **동적 공유**(`compute_dynamic_parameter_sharing`, merge_mode='majority')
  - 첫 측정: cluster label을 network로 쓴다.
  - cluster 수가 같으면: Hungarian 매칭.
  - 분할: 새 network에 겹침이 가장 큰(없으면 가장 가까운) 기존 network를 복사한다.
  - 병합: 번호를 다시 매기고 다수 network의 파라미터를 복사한다.
  - actor와 critic을 함께 복사한다.

**원 구현과 다른 점**
- **network 구조**: MAPSNet(MLP shallow+deep, LayerNorm, network별 초기화) 대신 공통 CNN+GRU를 쓴다. hetdps mode에서는 shallow/deep 배정이 같으므로 network 단위 배정과 동등하다.
- **학습기**: 비순환 MLP PPO(n_steps 5 × 32 env)가 아니라 공통 recurrent PPO를 쓴다.
- **측정 시점**: 3,200 env step 기준을 update 경계(1,024 step)에서 확인하므로 약 3 update마다 측정한다.
- **라이브러리**: scikit-learn/scipy가 없어 AP와 Hungarian을 numpy로 구현했다. Hungarian은 N≤4이므로 정확한 순열 탐색이다.
- **network 수 한계**: 분할로 network id가 N을 넘으면 비어 있는 index를 재사용한다. 원 코드는 이 경우 오류가 난다.
- **buffer 처리**: episode 경계 transition은 제외하고, 관측은 float16으로 저장한다.
- **생략한 것**: 측정 전용인 model-based 정책 거리(`compute_fusions`, `if_measure`)는 HetDPS 결정에 쓰이지 않으므로 계산하지 않았다.

**추가 비용**: 측정 1회 약 4.4초(GPU, VAE 537,915 parameters, 12,800 transitions). 15M step이면 4,687회로 약 5.8시간 추가되며, 학습 시간의 약 5%다. RAM은 약 80MiB다.

## 검증 (smoke, 각 64 step)

- **MAPPO**: 2회 update. approx_kl은 약 1e-9로 actor replay가 일치했다. critic replay 오차는 약 1.5e-7이다.
- **SePS**: 클러스터링 전 배정 [0,1,0,1] → VAE + DB → K=3, 배정 [2,0,2,1].
- **HetDPS**: 측정 2회, 배정 [0,1,0,0] → [0,0,0,1].
- **W&B**: 4개 run이 모두 `finished`(project `rware-from-scratch-smoke`), Step = env steps.
- **단위 테스트**: `tests/test_latent_team.py`의 군집·AP·Hungarian·동적 공유·ValueNorm·W2 테스트 포함, 전체 182 passed.
- 짧은 smoke에서 reward가 오르지 않은 것은 알고리즘 성능 판단 근거가 아니다.
