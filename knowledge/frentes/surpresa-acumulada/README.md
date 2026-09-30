# Frente: detector de surpresa acumulada

**Premissa:** [1 — "a quebra aparece em resumos"](../../premissas/01-surpresa-acumulada.md) ·
**Aberta em:** 2026-09-29 · **Fechada em:** 2026-09-29, **nula no S3** · **Branch:** `frente/surpresa-acumulada` · **Código:** `src/surpresa/`
(pacote separado; o `sbrt` fica intocado até o S4)

## Conclusão (2026-09-29)

**A surpresa acumulada funciona como detector, mas não soma ao Onyx.** Sozinha, fica no nível das
melhores colunas individuais do Onyx (0,575–0,584 contra 0,563–0,580). Combinada com o OOF, rende
+0,0001 [−0,0003; +0,0004]. O sinal dela é o eixo de variância, que o Onyx já cobre. O resto é
ruído, porque ~⅔ das quebras não aparecem em eixo simples nenhum
([premissa 4](../../premissas/04-gerador-das-quebras.md)). O código fica no repositório (`src/surpresa/`,
testado) para reaproveitar.

## Achado pós-fechamento (2026-09-29): o Onyx é quase cego a QUEDAS de variância

Mesma estratificação do [S1](S1-detector-sozinho.md), agora com o OOF do Onyx (`oof_b0_vema_bag4`).
Percentil médio no corte transversal do passo, m passos depois de τ
(script: scratchpad `onyx_por_tipo.py`):

| m | tipo (n em m=100) | **Onyx** | `sr_cal` | `fam_escala_sobe` | `fam_escala_desce` |
|---|---|---|---|---|---|
| 25 | desce | **0,488** | 0,560 | 0,350 | **0,696** |
| 100 | desce (209) | **0,560** | 0,638 | 0,297 | **0,804** |
| 200 | desce | **0,563** | 0,680 | 0,301 | **0,810** |
| 100 | sobe (230) | 0,828 | 0,734 | 0,840 | 0,264 |
| 100 | var_fixa (1052) | 0,561 | 0,505 | 0,550 | 0,444 |
| — | negativos | 0,471 | 0,482 | 0,481 | 0,510 |

- As quebras de **queda** de variância são tão comuns quanto as de alta (16,8% contra 15,9% das
  quebras). O Onyx as coloca no percentil **0,49–0,56**, quase no nível dos negativos (0,47). A
  família `escala_desce` as coloca em **0,70–0,81**.
- Mecanismo provável: os acumuladores mais fortes do Onyx olham só a cauda de cima (o
  `conformal_logm_abs` usa o p-value **superior** de |e|; o filtro bayesiano e o `cusum_var_up_*`
  crescem com a variância). O `cusum_var_down_r050` existe, mas não compensa.
- **Por que o S3 não pegou isso:** a combinação foi **aditiva** e usou o score agregado. A
  `escala_desce` sobe as quedas, mas derruba as altas (0,26) e os `var_fixa` (0,44), então somá-la
  linearmente se cancela, e o logsumexp dilui. Ela precisa entrar como **coluna na árvore**, que
  sabe condicionar ("evidência de queda alta **e** de alta baixa"), ou numa combinação
  não linear.
- **Correção de uma leitura do S1:** o Onyx põe as `var_fixa` em 0,56–0,59 com m ≥ 100, acima dos
  negativos. Elas são fracas nos 10 eixos do censo, mas **não invisíveis**: o Onyx extrai algo que o
  detector de surpresa não extrai.
- **Próximo experimento (S5, ainda não registrado):** as 8 colunas `fam_*` (ou só `escala_desce` e
  `escala_sobe`) entram como features do Onyx, com K=4 e R0 na partição 42. Bucket-alvo:
  `150<t≤400` e `t>400`, onde a separação das quedas é maior.

## A ideia em uma frase

Aprender o comportamento de cada série no histórico, levar cada ponto novo a um fluxo que sob H0 é
N(0,1) **em qualquer série**, e acumular razões de verossimilhança contra alternativas direcionais.
O resultado é um número de evidência com o mesmo significado em todas as séries.

## Desenho

### 1. Modelo por série (ajustado uma vez no histórico)

| Etapa | O quê | Por quê |
|---|---|---|
| Branqueamento | AR(p) por mínimos quadrados, com intercepto (p=10, como o Onyx) → resíduo `r` | tira a dependência linear do caminho; o que sobra é inovação |
| Fluxo A, **escala congelada** | `a_t = r_t / σ̄`, com σ̄ do histórico | variância e cauda. Não pode ter volatilidade adaptativa, senão absorve a quebra de variância (a trava CE2 do Onyx, [MODELO §3.4](../../modelo/MODELO.md)) |
| Fluxo B, **vol-ajustado** | `b_t = r_t / √v_t`, GARCH(1,1) com *variance targeting* em σ̄², (α, α+β) escolhidos numa grade pela quase-verossimilhança do histórico | dependência, média e forma, sem a confusão do clustering de volatilidade |
| PIT | CDF empírica do fluxo no histórico (interpolada), com cauda exponencial além dos quantis 2,5%/97,5% → `u_t` | forma arbitrária, não gaussiana |
| Universalização | `g_t = Φ⁻¹(u_t)` | sob H0, `g ~ N(0,1)` iid em qualquer série |

### 2. Evidência: alternativas direcionais com log-razão de verossimilhança exata contra N(0,1)

| Família | Fluxo | Alternativa | Incremento ℓ_t |
|---|---|---|---|
| `escala_sobe` | A | g ~ N(0, s²), s ∈ {1,2; 1,5; 2} | −log s + g²/2·(1 − 1/s²) |
| `escala_desce` | A | s ∈ {0,8; 0,65} | idem |
| `cauda` | A | t de Student de variância unitária, ν ∈ {4; 8} | log t̃_ν(g) − log φ(g) |
| `forma3`, `forma4` | A | densidade de PIT 1 + θ·L̃_k(u) (Legendre normalizado), θ = ±0,15 | log(1 + θ L̃_k(u)) |
| `media` | B | N(±δ, 1), δ ∈ {0,25; 0,5; 1} | ±δg − δ²/2 |
| `correlacao` | B | g_t \| g_{t−1} ~ N(ρ g_{t−1}, 1−ρ²), ρ ∈ {±0,15; ±0,3} | log da normal condicional − log φ(g_t) |
| `arch` | B | g_t ~ N(0, h_t), h_t = max(0,05; 1 + a(g²_{t−1} − 1)), a ∈ {0,2; 0,4} | −½log h_t − g²/(2h_t) + g²/2 |

Cada alternativa k roda um **Shiryaev-Roberts** em log (`log R_t = logaddexp(0, log R_{t−1}) + ℓ_t`),
que soma a evidência sobre todos os instantes de início possíveis, e um CUSUM
(`W_t = max(0, W_{t−1} + ℓ_t)`). Sob H0, `E[R_t] = t` em qualquer série: a deriva é comum a todas e,
pela invariância C1, neutra na ordenação.

**Saídas por passo:**
- `sr`: logsumexp_k log R_k − log K, o **score direcional**;
- `sr_cal`: o mesmo com incrementos corrigidos pelo viés medido fora da amostra, ver abaixo;
- `ingenua_z`: Σ(g_A² − 1)/2 / √(t/2), a surpresa somada do texto original, como **controle**;
- `cusum`: max_k W_k;
- uma coluna por família (máximo sobre as magnitudes, versão temperada `sr_eta`), para servir de feature no S4.

### 3. Calibração honesta por série (`sr_cal`)

O modelo é ajustado nos primeiros ⅔ do histórico e aplicado ao ⅓ final, que é H0 fora da amostra.
Para cada alternativa k, mede-se a média do incremento ℓ_k nesse trecho e compara-se com o valor
teórico sob N(0,1), que é −KL. A diferença é o viés da série naquela direção, causado por erro de
especificação ou de estimação. Online, os incrementos são descontados desse viés, e o modelo usado
é o ajustado no histórico inteiro. Nenhum rótulo entra.

### 4. Implementação

- `src/surpresa/modelo.py`: `ajustar(hist, cfg) -> ModeloSerie` (imutável).
- `src/surpresa/evidencia.py`: alternativas, incrementos vetorizados e recursão SR/CUSUM.
- `src/surpresa/stream.py`: `pontuar_serie(hist, online, cfg)` em lote (usado na avaliação) e
  `SurpresaStream` passo a passo (o que vira `StateBlock` no S4). Um teste exige que os dois
  concordem.
- `configs/surpresa.yaml`: todos os números.
- `scripts/surpresa_avaliar.py`: 10k séries → `artifacts/surpresa/*.parquet` + TS-AUC do board.

## Experimentos

| ID | Ficha | Status |
|---|---|---|
| B0 | [Reconstruir o incumbente neste clone](B0-incumbente.md) | concluído: 0,6278 (baseline relativo) |
| S0 | [Sanidade sintética](S0-sanidade-sintetica.md) | concluído: rodada 2 passa com `sr_eta` (têmpera por direção) |
| S1 | [Detector sozinho nas 10k séries](S1-detector-sozinho.md) | concluído: **0,575** [0,567; 0,583]; sinal só nas quebras de variância (~⅓) |
| S2 | [Redundância transversal contra o OOF do Onyx](S2-redundancia.md) | concluído: ρ = 0,28 com o OOF (pouca redundância) |
| S3 | [Combinação sem retreino + R0](S3-combinacao.md) | **nulo**: +0,0001 [−0,0003; +0,0004] |
| S4 | `StateBlock` no Onyx + K=4 + R0 nas duas partições | **não executado** (S3 nulo) |
| S5 | [Famílias `escala_desce`/`escala_sobe` como features do Onyx](S5-familias-como-features.md) | rodando (aberto pelo achado das quedas de variância) |
