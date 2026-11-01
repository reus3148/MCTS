---
title: 최종 발표 덱 — 장별 수치·출처
updated: 2026-11-01
---

# 최종 발표 덱 — 장별 수치·출처

덱(Claude 아티팩트 `MCTS-ONC 최종 발표`, https://claude.ai/artifact/4c3wRbGxQH4Ac52KFNz9nc)은
**화면에 출처와 설명 문장을 넣지 않습니다.** 설명은 각 장의 발표자 노트에 있고, 화면에
나온 숫자가 어디서 왔는지는 이 문서에 모았습니다. 질의응답 때 이 문서를 옆에 둡니다.

- 내부 수치는 모두 커밋된 `reports/<label>/metrics.json`·`tables/`에서 왔습니다. 손으로
  옮긴 값이 아닙니다.
- 덱의 숫자를 바꾸면 **이 문서와 `docs/final-claims.md`를 같이** 고칩니다.
- 장 번호는 2026-11-01 재구성(본편 18장 + 부록 7장) 기준입니다.

---

## 본편

| # | 장 (id) | 화면의 수치·주장 | 출처 |
|---|---|---|---|
| 1 | 표지 (`cover`) | — | — |
| 2 | 알파고 (`alphago`) | 2016년 3월 서울, 4 : 1 · 바둑 국면 수 약 10¹⁷⁰ | Silver D, et al. *Mastering the game of Go with deep neural networks and tree search.* Nature 2016;529:484–489 · 국면 수: Tromp J, *The number of legal Go positions* (2016), ≈2.08×10¹⁷⁰ · 바둑판 그림은 설명용(실제 기보 아님) |
| 3 | 딥블루·알파제로 (`deepblue`) | 1997 딥블루 3.5 : 2.5, 초당 2억 수 · 2017 알파제로 초당 6만 수, 스톡피시 6천만 | Campbell M, Hoane AJ, Hsu F. *Deep Blue.* Artif Intell 2002;134:57–83 (1997 대국 실측 평균 1.26억 수/초, 최대 3.3억) · IBM History "Deep Blue" (2억 수/초, 3.5–2.5) · Silver D, et al. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science 2018 (6만 대 6천만, 1,000판 155승 6패 839무) |
| 4 | 엔진 이후의 정석 (`openings`) | 루이 로페즈 1.e4 e5 2.Nf3 Nc6 3.Bb5 · 퀸스 갬빗 1.d4 d5 2.c4 | 표준 오프닝 수순 · 연대(16세기 루이 로페즈, 15세기 필사본의 퀸스 갬빗)는 노트에만 있고 **원문 대조 안 함** |
| 5 | 연구 질문 (`question`) | — | `유방암_RL_연구제안서.md`, `src/minutes/2026-05-26-kickoff.md` |
| 6 | 체스판과 치료판 (`chess`) | 행동축 5개 · 확률 전이 | `configs/dynamic_v0_5.json`, `reports/dynamic-mcts-poc-v0.2` (chance-aware UCT) |
| 7 | 선행연구 (`prior-work`) | Watson for Oncology · LLM 종양위원회 · 암 임상시험 Q-learning · AI Clinician · 의료 강화학습 지침 | 아래 "외부 문헌" 1–6 |
| 8 | 설계 (`setup`) | 치료 5축 · 예산 1024 · METABRIC Cox C-index 0.677 · 환자 40명 · 시드 12 | `reports/mcts-poc-v1` (C-index OS 0.677 · RFS 0.649), `reports/budget-scaling-v0.4` (예산 1024 근거), `reports/sensitivity-patients-v1.1`·`reports/cohort-replication-v1.2` (40명·12시드) · NCCN 규칙: `docs/nccn-policy-spec.md` |
| 9 | 첫 결과 (`first-result`) | +0.0332 (95% CI +0.0271 ~ +0.0393) | `reports/cohort-replication-v1.2` · 예산에 따라 격차가 커진다(+0.0212 → +0.0314): `reports/budget-scaling-v0.4` |
| 10 | 네 단계 (`method`) | 81% · 44.5% · 1 : 3 · 0 / 5 | 각 단계 장 참조 · 절차 전체: `reports/procedure-checklist` (fig52) |
| 11 | 01 판이 공정한가 (`channels`) | 40%(z = −5.37) · 41%(z = −11.9) · 둘 다 끄면 +0.0013 (허용치 0.005) | `reports/reward-confounding-v1.4`, `reports/channel-decomposition-v1.5` · 그림 fig40 |
| 12 | 02 다이얼 (`tipping`) | 전환점 44.5% · 문헌값에서 −0.0061 (SE 0.0012) · 전환점 위험비 항암 0.880 · 내분비 0.797 · 방사선 0.734 (노트) | `reports/benefit-tipping-v2.1` (`metrics.json`, 강도별 표) · **끝점 위험비는 EBCTCG 자릿수로 적은 선언값, 원문 대조 대기** (`docs/human-review.md`) |
| 13 | 02 긴 게임 (`tipping-v07`) | NCCN 대비 +0.7~1.2% (강도 0.30~1.00) · 최선 고정 계획 여지의 70%가 강화 단 몫 (강도 0.445) | `reports/tipping-v07-v2.6` · 그림 fig54 · 분해는 **사후·정적** |
| 14 | 03 기보와 대조 (`real-care`) | 우리 판 1회·20.8% · 실제 3회·66% (진단 90일 내 진입 1,669명) · 가린 효과 −0.00037 · 진행 뒤 치료 변경 2.41배 / 2.42배 | `reports/msk-chord-profile-v2.2` (적응 기회·MSK 2.41배), `reports/genie-bpc-profile-v1.6` (GENIE 2.42배), `reports/closed-loop-value-v1.7` (−0.00037) · MSK-CHORD: 외부 문헌 7 · 보정하지 않은 연관 |
| 15 | 04 탐색 전에 지도 (`five-tries`) | v1.8 판돈 · v1.9·v2.0 상태 의존 · v2.4 해상도 · v2.5 잡음의 87% | `reports/decision-points-v1.8`, `reports/horizon-v1.9`, `reports/indifference-v2.0`, `reports/burden-map-v2.4`, `reports/stakes-map-v2.5` |
| 16 | 적용 결과 (`verdict`) | 81% · 44.5% · 70% · "최적 증명이 아니라 반증 실패" | 11·12·13장과 같음 · 문구 경계: `docs/final-claims.md` D절 |
| 17 | 한계 (`limits`) | 등급 1 한계 4건 · E-value 1.59 | `docs/limitations.md` (26건 + 철회 4건) · E-value: `reports/ipw-target-trial-v0.6` |
| 18 | 마무리 (`close`) | K-CURE · 위암·폐암 · 엔진/질환 분리 | `docs/k-cure-variable-dictionary.md`, `analysis/dynamic/schema.py` |

## 부록

| 장 (id) | 화면의 수치 | 출처 |
|---|---|---|
| A 교정 후 격차 (`headline`) | +0.0199 · +0.0136 · 3.7배 · 44.5% 미만 · 1/40 음수 | `reports/reward-confounding-v1.4`, `reports/benefit-tipping-v2.1`, `docs/results-reconciliation.md` · 인용 규칙: `src/minutes/2026-09-05-headline-sampling-decision.md` · 그림 fig23 |
| B 다이얼 곡선 (`tipping-evidence`) | 곡선 · 75%가 강도 0.25에서 사라짐(z = −10.2) · NCCN +0.0609 / MCTS +0.0350 · 항암 수용 31% → 72% (노트) | `reports/benefit-tipping-v2.1` · 그림 fig47 |
| C 눈 가리기 (`closed-loop`) | −0.00037 (z = −0.29) · 검출 한계 0.00359 | `reports/closed-loop-value-v1.7` · 그림 fig43 |
| D ECOG (`ecog`) | 2.55배 · +1.3%p (한계 3.0%p) · ECOG≥2 항암 +8.1%p | `reports/ecog-state-v2.3` · 그림 fig49 |
| E 사다리 (`shape`) | 판돈 ½ (0.0757 → 0.0392) · 25배 (0.00070 → 0.01744) · 잡음 0.02의 87% | `reports/stakes-map-v2.5`, `reports/burden-map-v2.4` · 그림 fig51 |
| F 체크리스트 (`checklist`) | 진단 5 · 습관 3 · 사람 검토 29건 중 2건 | `reports/procedure-checklist`, `docs/human-review.md` · 그림 fig52 |
| G 주장하지 않는 것 (`not-claim`) | 5항목 | `docs/final-claims.md` D절 |

## 그림 원본

덱의 그림은 아티팩트 자산으로 올라가 있고, 원본은 저장소에 있습니다.

| 덱 장 | 그림 | 저장소 |
|---|---|---|
| 01 판이 공정한가 | fig40 | `public/figures/fig40_channel_decomposition.png` |
| 02 긴 게임 | fig54 | `public/figures/fig54_static_decomposition_v07.png` |
| 부록 A | fig23 | `public/figures/fig23_results_reconciliation.png` |
| 부록 B | fig47 | `public/figures/fig47_benefit_tipping.png` |
| 부록 C | fig43 | `public/figures/fig43_closed_loop_value.png` |
| 부록 D | fig49 | `public/figures/fig49_ecog_state.png` |
| 부록 E | fig51 | `public/figures/fig51_stakes_map.png` |
| 부록 F | fig52 | `public/figures/fig52_procedure_checklist.png` |

fig48(MSK-CHORD 적격성)은 재구성 뒤 덱에서 빠졌습니다. 03장 질의 때 `public/figures/fig48_msk_chord_profile.png`를 띄웁니다.

## 외부 문헌 (선행연구·자료)

1. Somashekhar SP, et al. **Watson for Oncology and breast cancer treatment recommendations: agreement with an expert multidisciplinary tumor board.** *Ann Oncol* 2018;29(2):418–423. — 유방암 638명, 일치율 93%
2. Sorin V, et al. **Large language model (ChatGPT) as a support tool for breast tumor board.** *npj Breast Cancer* 2023;9. doi:10.1038/s41523-023-00557-8 — 10명 중 7명 일치
3. Murphy SA. **Optimal dynamic treatment regimes.** *J R Stat Soc B* 2003;65:331–355. — 노트에서만 언급
4. Zhao Y, Kosorok MR, Zeng D. **Reinforcement learning design for cancer clinical trials.** *Stat Med* 2009;28(26):3294–3315.
5. Komorowski M, et al. **The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care.** *Nat Med* 2018;24:1716–1720.
6. Gottesman O, et al. **Guidelines for reinforcement learning in healthcare.** *Nat Med* 2019;25:16–18.
7. Jee J, et al. **Automated real-world data integration improves cancer outcome prediction.** *Nature* 2024. PMID 39506116. (MSK-CHORD, CC BY-NC-ND 4.0 — 집계값만 인용)
8. AACR Project GENIE Consortium. **GENIE BPC Breast Cancer v1.0-public.** (집계값만 인용)
9. Curtis C, et al. **The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups.** *Nature* 2012. (METABRIC)

1–6은 논문 서론 §1.2의 "관련 연구 10~15편" 후보이기도 합니다(`docs/paper-draft.md`, 아직 반영하지 않음).
