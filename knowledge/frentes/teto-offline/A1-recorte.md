# A1 — Aumento de dados por re-corte de séries reais

**Frente:** [teto offline](README.md) · **Status:** concluído: **piora** · **Hipótese:** 2026-09-30 00:50 · **Resultado:** 01:45

## Ideia

Cada série real é a concatenação S = histórico ++ online, com a quebra (se houver) no índice
b = n_h + τ. Uma série **nova** sai de um novo ponto de corte c dentro da região H0 (c ≤ b), com um
novo histórico S[c−n_h' : c] e um novo online S[c : c+n_on']. Os sorteios reproduzem a estrutura da
competição: n_h' ~ U[1000, 5000] (limitado ao disponível), n_on' ~ U[10, 999], τ' ~ U[0, n_on'). Tudo
é re-padronizado pela média e dp do novo histórico, como faz a plataforma. **Todos os valores são
reais, e as quebras são as do gerador.** O que muda é τ, o histórico e o comprimento: cada quebra
aparece em novas posições e contra novos históricos.

**Limite honesto:** o re-corte não cria eventos de quebra novos (continuam 4.967). O C1 mediu o ganho
de ×2 **séries**; se ele vier da diversidade de eventos, o re-corte rende menos que isso.

## Protocolo sem vazamento

- As séries aumentadas recebem `id = 10000·k + id_origem`.
- A divisão em folds é a do Onyx, calculada **só nas séries originais**. Cada série aumentada entra
  no **treino** do fold cuja série de origem está no treino, e **nunca** na validação.
  Implementado por monkeypatch de `grouped_stratified_kfold` num wrapper do `train.py`, sem editar
  o `sbrt`.
- O OOF é avaliado só nas séries originais, contra o B0 (mesmas sementes, mesma receita), com R0.

## Hipótese (escrita ANTES de medir)

- **A1 com k=1** (×2 de linhas): Δ entre **+0,003 e +0,010** contra o B0. Menos que os +0,0144 do
  C1, porque não há eventos novos.
- **Regra:** IC excluindo 0 → escalar para k=2–3 e medir a curva. Δ ~0 → o ganho do C1 vem de
  **eventos**, e o caminho passa a ser gerar eventos (simulação calibrada no censo,
  [premissa 4](../../premissas/04-gerador-das-quebras.md)).

## Resultado

**Rodada com 10k + re-cortes: abortada.** Com 4,68M linhas × 197 colunas e `linear_tree` (que guarda
os dados crus), o processo chegou a 18 GB de memória privada numa máquina de 16 GB, trocou para disco
(2.000 páginas/s, 0,5 s de CPU a cada 10 s) e levaria ~3 h. Pegadinha registrada abaixo.

**Rodada A1-metade** (`scripts/run_a1_metade.sh`): as 5k séries de id par + os re-cortes k=1 delas
(1,27M + 1,07M linhas), sementes 777 e 101, avaliadas nas mesmas séries do [C1](C1-curva-aprendizado-onyx.md):

| Treino | TS-AUC (séries de id par, grade do board) |
|---|---|
| 5k séries | 0,6129 |
| **5k + re-cortes** | **0,6008** |
| 10k séries | 0,6273 |

Por semente: 777 −0,0087 · 101 −0,0128.

## Decisão

**Descartado: o re-corte piora (−0,012).** A explicação mais provável é que ele **duplica eventos**: cada
quebra real aparece de novo com features muito correlacionadas, o que dobra o peso das
idiossincrasias de cada evento. O modelo decora mais, e a variância sobe em vez de cair. **Isso
confirma que o gargalo do C1 é o número de eventos independentes, não de vistas.**

**Pegadinha operacional (vale para qualquer aumento):** acima de ~2,5M linhas com `linear_tree`, o
treino não cabe em 16 GB. Meça em metade das séries ou desligue o `linear_tree` no braço.
