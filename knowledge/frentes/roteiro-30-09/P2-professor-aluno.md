# Passo 2 — Professor e aluno (rótulos de detectabilidade por linha)

**Frente:** [roteiro 30/09](README.md) · **Status:** concluído: **descartado** (principal); sucedido pela frente [oráculo destilado](../oraculo-destilado/README.md) · **Hipótese:** 2026-09-30 11:30

## Por quê

O Onyx é limitado por quebras ([C1](../teto-offline/C1-curva-aprendizado-onyx.md)), e parte dessa
escassez vem de **rótulos ruidosos**. Uma linha "quebrou" dois passos depois de τ, ou numa quebra
invisível, não carrega informação, e o modelo gasta eventos tentando ajustá-la. O X1-soft (peso por
série segundo a detectabilidade) rendeu +0,004. O professor é a versão por **linha**.

## Desenho

**Professor** (`scripts/p2_professor.py`), com os folds do Onyx (`grouped_stratified_kfold`, `cfg.seed`):
- linhas numa grade m ∈ {2,3,5,8,12,20,30,50,80,120,200,300,500,800}, limitada ao pós disponível;
  negativos com pseudo-τ ~ U[0, len−1] e a mesma grade;
- features privilegiadas: as 118 de `src/offline/features.py` sobre [τ, τ+m) (prefixo `a_`), as 118
  sobre o pós **inteiro**, o futuro (prefixo `f_`), e log m;
- alvo: positivo × negativo. A probabilidade = chance de a quebra já ser perceptível em m;
- **nested cross-fit:** para cada fold f do aluno, o professor é ajustado só nas séries fora de f, com
  CV interna de 4 partes (por série) gerando a nota dessas séries. As séries de f nunca passam pelo
  professor que rotula o treino de f. Saída: `y_soft_f{f}` por linha do `train_rows`, interpolada em
  log m (m=1 recebe o valor de m=2). Linhas pré-quebra e negativos ficam com 0.

**Aluno:** `LightGBMConfig.soft_label` (default `False`). Ligado, o fold k treina com
`0,5·y + 0,5·y_soft_fk` e `objective="cross_entropy"`. Pesos, curva de taxa-base, feval e OOF continuam
no `y` verdadeiro. Receita do E1.

## Hipótese e regras (escritas ANTES de medir)

- Δ contra o E1: **+0,004 a +0,012**.
- **Adotar** se ≥ +0,005 com IC excluindo 0 na partição 42 (K=4) **e** na 43 (K=2). **Descartar** se
  ≤ +0,002.
- Uma mistura principal (0,5/0,5); outras seriam exploratórias.
- Sanidade: com `soft_label=False`, o treino é o E1 exato (no-op). Professor com AUC por linha bem
  acima do aluno (ele conhece τ e o futuro).
- Limite declarado: mesmo aprovado hoje, a produção precisa do professor dentro do `train()` da
  nuvem. É trabalho separado, não incluído neste passo.

## Resultado

Professor (partição 42): AUC por linha **0,667–0,672** nos 5 folds (interno); notas nas linhas positivas
com mediana 0,46 (quantis 5%/95%: 0,00/0,92).

Aluno, mistura 0,5/0,5 (`p2_aluno`), R0 do bag de 4 com v-EMA contra o E1:
**0,6339 contra 0,6349, Δ −0,0010 [−0,0029; +0,0011]**. Por semente foi positivo nas 4 (+0,0002, +0,0016,
+0,0007, +0,0027), mas o ganho **some no bag**, o mesmo padrão do V7: redução de variância que o bag já
faz. `cross_entropy` + `linear_tree` funcionou.

## Decisão

**Descartado** (≤ +0,002). O braço exploratório "nota como peso" foi interrompido (prior baixo) para
liberar RAM para a [frente do oráculo destilado](../oraculo-destilado/README.md), que corrige os três
defeitos deste desenho: um professor que conhecia τ, negativos presos em 0 e metade de rótulo duro.
