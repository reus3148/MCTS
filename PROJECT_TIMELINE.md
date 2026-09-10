# MCTS-ONC 프로젝트 타임라인

나중에 제3자가 연구의 순서와 근거 파일을 빠르게 확인할 수 있도록 날짜별
산출물을 연결한 인덱스입니다. 세부 의사결정은 각 Minutes 원문에 남깁니다.

| 날짜 | 이정표 | 실제 산출물 | 상태 |
|---|---|---|---|
| 2026-05-26 | 연구 킥오프와 범위 확정 | `src/minutes/2026-05-26-kickoff.md` | 완료 |
| 2026-05-27 | METABRIC 전처리·기초 시각화 | `patients.csv`, Figure 01~07, `analysis/02~03` | 완료 |
| 2026-05-27 | 단순화 NCCN 정책 A와 일치율 | `patients_with_nccn.csv`, Figure 08~09, `analysis/04~05` | 완료 |
| 2026-05-27 | 연구 블로그와 Minutes 체계 | `/minutes`, 날짜별 Markdown 기록 | 완료 |
| 2026-07-11 | Cox 보상모형 + UCT-MCTS PoC v0.1 | `analysis/mcts`, `analysis/06~07`, Figure 10~13 | 완료 |
| 2026-07-11 | 확률적 동적환경 + MCTS PoC v0.2 | `analysis/dynamic`, `analysis/08~09`, Figure 14~17 | 완료 |
| 2026-07-11 | K-CURE 이행 데이터 계약 | `docs/k-cure-adaptation.md` | 초안 완료 |
| 2026-08-24 | 다중 시드 강건성 분석 v0.3 | `analysis/10`, `reports/robustness-v0.3` | 완료 |
| 2026-08-24 | 합성 가정 민감도 분석 v0.3 | `analysis/11`, `reports/sensitivity-v0.3` | 완료 |
| 2026-08-24 | Target trial 인과 명세 초안 | `docs/target-trial-protocol.md` | 초안 완료 |
| 2026-08-24 | K-CURE 변수 요청·매핑 명세 | `docs/k-cure-variable-dictionary.md` | 초안 완료 |
| 2026-08-26 | 탐색 예산 스케일링 진단 v0.4 | `analysis/12`, `reports/budget-scaling-v0.4` | 완료 |
| 2026-08-27 | MCTS·MDP 개념 감사와 환경 편향 수정 v0.5 | `analysis/13`, `configs/dynamic_v0_5.json`, `reports/environment-fix-v0.5` | 완료 |
| 2026-08-27 | 발표용 자료 정비 — Figure 18~23·이야기·숫자화해·제안서대조 | `analysis/14`, `docs/research-story.md`, `docs/results-reconciliation.md`, `docs/proposal-vs-delivered.md`, `/story` | 완료 |
| 2026-08-27 | IPW 표적시험 에뮬레이션 v0.6 | `analysis/causal`, `analysis/15~16`, `reports/ipw-target-trial-v0.6`, Figure 24~26 | 완료 |
| 2026-08-27 | 이중강건 추정·결정별 식별 가능성 지도 v0.7 | `analysis/17~18`, `reports/doubly-robust-v0.7`, Figure 27~28 | 완료 |
| 2026-08-28 | v0.5 환경 재실행 검증 | `reports/*-v0.5env`, Figure 30 | 완료 |
| 2026-08-28 | 2차원 상호작용 민감도 v0.8 | `analysis/19~20`, `reports/interaction-sensitivity-v0.8`, Figure 29 | 완료 (일부 정정됨) |
| 2026-08-28 | 가치판단 상호작용 확인 v0.9 | `analysis/22~23`, `reports/utility-interaction-v0.9`, Figure 31 | 완료 |
| 2026-08-28 | 민감도 정밀도 재실행 v1.0 | `analysis/24~25`, `reports/sensitivity-precision-v1.0`, Figure 32 | 완료 (v0.3 일부 철회) |
| 다음 단계 | 인과추정 시제품·상호작용 민감도·K-CURE 확보 | IPW/g-방법, 2D 민감도, utility 사전등록, 코드북 매핑, 임상 검토 | 예정 |

## 2026-07-11 기준 현재 위치

```text
METABRIC 원본
  -> 전처리 환자표
  -> NCCN 정책 A
  -> 생존 보상모형
  -> 4단계 치료환경 (HR 적격성 반영, 환자별 8~16개 경로)
  -> UCT-MCTS 정책 B
  -> 보류 테스트셋 비교와 완전탐색 검증 (v0.1)
  -> OS/RFS 기반 5년 확률환경 (v0.2)
  -> 환자별 18~135개 의사결정 궤적
  -> 반응 후 재계획하는 stochastic MCTS

다음: 합성 전이 -> K-CURE 실제 추정, 관찰 연관성 -> 인과효과 추정
```

MCTS v0.1의 수치·방법·한계는
[기술 보고서](reports/mcts-poc-v1/README.md)와
`src/minutes/2026-07-11-mcts-poc-v1.md`에서 확인할 수 있습니다.

동적 v0.2는 [기술 보고서](reports/dynamic-mcts-poc-v0.2/README.md)와
`src/minutes/2026-07-11-dynamic-environment-v02.md`에서 확인할 수 있습니다.

v0.4 탐색 예산 진단은 [기술 보고서](reports/budget-scaling-v0.4/README.md)와
`src/minutes/2026-08-26-budget-scaling-v04.md`에서 확인할 수 있습니다.

v0.5 환경 감사는 [기술 보고서](reports/environment-fix-v0.5/README.md)와
`src/minutes/2026-08-27-environment-audit-v05.md`에서 확인할 수 있습니다.
**새 실험은 `configs/dynamic_v0_5.json`을 씁니다.** v0.2~v0.4 리포트 수치는
편향이 있던 환경에서 나온 것이며, 아직 v0.5로 재실행하지 않았습니다.
이 진단 이후 **기본 탐색 예산은 1024 이상**을 쓰며, 256으로 낸 v0.2·v0.3의
결정 단위 수치는 탐색 해상도 미달이라는 단서와 함께 읽어야 합니다.

v1.0 민감도 정밀도 재실행은 [기술 보고서](reports/sensitivity-precision-v1.0/README.md)와
`src/minutes/2026-08-28-sensitivity-precision.md`에서 확인할 수 있습니다.
이 진단 이후 **민감도의 최소 설계는 시드 12개·예산 1024**입니다.

v1.1 환자 수 민감도는 [기술 보고서](reports/sensitivity-patients-v1.1/README.md)와
`src/minutes/2026-09-04-sensitivity-patients-v11.md`에서 확인할 수 있습니다.
환자 8 → 20명에서 헤드라인은 +0.0004만 움직였지만, 환자 부트스트랩에서
"상위 2개가 가치판단"의 재현율이 **8명 24.6% → 20명 69.5%** 로 나타나
**최소 설계에 "환자 20명"이 추가**됐습니다. 8명으로 낸 순위는 인용하지 않습니다.

v1.2 코호트 복제는 [기술 보고서](reports/cohort-replication-v1.2/README.md)와
`src/minutes/2026-09-04-cohort-replication-v12.md`에서 확인할 수 있습니다.
겹치지 않는 20명 두 개로 순위를 재검했고, **실행 전에 선언한 예측이 빗나갔습니다**
(순위 Spearman +0.03). 순위는 **합산 40명으로만, 그룹 수준으로만** 말합니다.
사후 분석에서 **아형 균등 표집이 헤드라인을 41% 부풀리고 있었음**을 찾았습니다
(+0.0332 → 실제 아형 구성 표준화 시 +0.0198).

v1.3 음성대조와 트리밍 수정은 [기술 보고서](reports/endocrine-effect-v1.3/README.md)와
`src/minutes/2026-09-05-endocrine-negative-control.md`에서 확인할 수 있습니다.
호르몬치료가 들을 수 없는 ER 음성 환자에서 ER 양성과 거의 같은 위험비가 나와
**음성대조가 실패**했고, 이후 관찰 추정치는 "효과"가 아니라 **"교란 보정 후에도 남는
연관성"** 으로 적습니다. 같은 작업에서 **트리밍이 v0.6부터 고정점에 도달하지 않고
있었음**을 찾아 고쳤고, 그 결과 v0.7의 "방사선 식별 불가" 판정이 철회됐습니다.

v1.4 보상모형 교란 진단은 [기술 보고서](reports/reward-confounding-v1.4/README.md)와
`src/minutes/2026-09-07-reward-model-confounding.md`에서 확인할 수 있습니다.
인과 갈래와 시뮬레이터 갈래를 처음으로 대조했더니, Cox 보상모형이 항암·호르몬치료를
**해롭다고** 믿고 있었고(우리 보정 분석과 부호 반대), MCTS가 NCCN과 갈린 모든 방향이
그 계수를 따라가고 있었습니다. **치료 계수를 중립화하자 격차의 40%가 사라졌고**
(z = −5.37), 사라진 이유는 MCTS가 나빠져서가 아니라 **NCCN이 벌점을 그만 받아서**
였습니다. 이후 새 실험은 치료 중립 보상모형을 쓰고, 인용 헤드라인은 **+0.0199**
(아형 표준화까지 하면 +0.0136)입니다.

v1.5 채널 분해는 [기술 보고서](reports/channel-decomposition-v1.5/README.md)와
`src/minutes/2026-09-07-channel-decomposition.md`에서 확인할 수 있습니다.
v1.4가 남긴 +0.0199를 선언된 이득 × 선언된 비용 2×2로 분해했더니, 영대조는 통과했고
(+0.0013) **비용을 없앤 것만으로 격차의 69%가 사라졌습니다**(z = −11.9). 원인은
config에 있었습니다 — **표준 항암·표준 호르몬·국소 방사선은 선언된 생존 이득이 0인데
비용만 있고**, 그 셋이 NCCN이 처방하는 것들입니다. 원래 +0.0332의 **81%가 두
비대칭**에서 나왔으며, 현재 환경에서는 "MCTS가 NCCN보다 낫다"를 주장하지 않습니다.

v1.6 GENIE BPC 자료 적격성 평가는 [기술 보고서](reports/genie-bpc-profile-v1.6/README.md)와
`src/minutes/2026-09-10-genie-bpc-movetext.md`에서 확인할 수 있습니다.
팀이 확보한 외부 자료 안에 **GENIE BPC 유방암 v1.0-public**(실제 환자 1,130명 · index
레지멘 6,579건 · 영상 판독 26,773건)이 있었습니다. 2026-05-26 킥오프에서 프레임을 좁힌
이유였던 **치료 시점 정보** — 체스 비유로 말하면 **기보** — 가 처음 손에 들어온 것입니다.
효과를 추정하기 전에 자료 적격성부터 쟀고, 실행 전에 예측 넷을 코드에 적었습니다.

**폐루프 적응은 기록에 남아 있었습니다.** 영상 판독이 "진행/악화"인 뒤 90일 안에 새
레지멘이 시작된 비율이 **71.3%**, "안정/호전" 뒤에는 **29.5%** — 2.41배입니다. 그리고
판독 **이전** 90일로 같은 계산을 하면 33.7% 대 30.8%로 거의 같아 **시간 위약대조를
통과**했습니다(v1.3의 음성대조 실패가 이 설계를 미리 넣게 만들었습니다).

동시에 한계가 분명합니다. **우리 환경의 행동축 5개 중 2개(항암·내분비)만 관측
가능**하고 수술·방사선·선행보조 시점은 큐레이션되어 있지 않으며, **ECOG·동반질환도
없습니다.** 그리고 사전 예측 하나가 빗나갔습니다 — 5년 내 레지멘 수 중앙값이 4.0으로,
**우리 환경의 행동축 개수가 진행성 질환에는 짧다**는 것이 드러났습니다.
v1.5의 최우선 과제는 사라지지 않고 **좁아졌습니다.**

같은 폴더에 K-CURE data free box의 **유방암 표준 테이블 가짜 데이터**(7개 병원)도 있어,
`BRST_DG_THNF.ECOG_CD`·`BRST_PT_HLNF.MHIS_*_YN` 등 실제 컬럼명이 확인됐습니다.
값은 난수이므로 분석용이 아니라 **어댑터 배관 검증용**입니다.
