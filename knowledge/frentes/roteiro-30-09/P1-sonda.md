# Passo 1 — Sonda dentro de cada série

**Frente:** [roteiro 30/09](README.md) · **Status:** concluído: **faixa do meio (0,578), passo 3 sai** · **Hipótese:** 10:50 · **Resultado:** 11:20

## Pergunta

As quebras que o censo chamou de invisíveis (`var_fixa`, ~⅔) mudam **algo**, em qualquer eixo? O T1 e
o T2 aprenderam **entre** séries, onde há poucos eventos. Aqui cada série treina o próprio
classificador, **dentro** dela, onde os dados são abundantes.

## Desenho (`scripts/k1_sonda_intra_serie.py`)

- **Amostra:** quebras com ≥ 400 pontos pós-τ; negativos com online ≥ 400, pseudo-τ ~ U[0, len−400]
  (a regra corrigida do T1). Tipos de quebra pela log-razão de variância do pós contra o histórico,
  como no S1: `sobe` > 0,3, `desce` < −0,3, `var_fixa` no meio.
- **Janelas de 20 pontos:** "antes" = últimos 800 pontos do histórico (passo 2); "depois" = [τ, τ+400)
  (passo 1). Validação **por blocos**: os primeiros 75% de cada trecho no treino e os últimos 25% no
  teste, com folga de 20 pontos.
- **Features por janela:** 20 valores ordenados, média, média de |x|, média de x², médias de x·x₋ₖ
  (k = 1..3). Padronizadas pelo treino da própria série.
- **Classificadores:** logística (C=1) e `HistGradientBoosting` (profundidade 3, 50 iterações), este
  com os 20 valores crus a mais. Nota = AUC nas janelas de teste; nota final = a maior das duas. O
  viés de "máximo de dois" é o mesmo em positivos e negativos, então a comparação entre grupos segue
  justa.
- **Sanidade:** 500 negativos recebem uma mudança **só de dependência** no trecho pós, recolorido com
  AR(1) φ=0,3 com variância preservada: yₜ = φyₜ₋₁ + √(1−φ²)·xₜ.

## Hipótese e critérios (escritos ANTES de medir)

- **Sanidade (a):** quebras de variância (`sobe` ∪ `desce`) contra negativos, AUC ≥ **0,75**.
- **Sanidade (b):** negativos com a dependência injetada contra os mesmos negativos intactos, AUC ≥ **0,75**.
- Se qualquer sanidade falhar, a sonda está quebrada e o critério principal não vale.
- **Critério principal:** `var_fixa` contra negativos. **≥ 0,60 → o passo 3 entra; ≤ 0,53 → o passo 3
  sai;** no meio, olhar quais grupos de features carregam a separação.
- Previsão pessoal: 0,53–0,57. O censo de 10 eixos já via um excesso fraco em dependência de curto
  prazo (7–9% contra 5%).

## Mudança de desenho antes de medir (sanidade no sintético)

O split único 75/25 deixava ~5 janelas independentes no teste, e o nulo sintético saía 0,65 numa
série. Troquei por **CV de 4 blocos contínuos** em cada trecho (folga de 20 pontos, AUC média entre
blocos). Nulo sintético: 0,494 ± 0,107 por série; variância ×1,2: 0,84; dependência injetada: 0,74.

## Resultado

`python -u scripts/k1_sonda_intra_serie.py --n-jobs 4` (~6 min): 3.036 negativos, 899 `var_fixa`, 180
`sobe`, 128 `desce`, 500 injetados. `artifacts/roteiro/p1_sonda/`.

| Teste | AUC | Critério |
|---|---|---|
| Sanidade (a): `sobe`∪`desce` × negativos | **0,793** | ≥ 0,75 ✓ |
| Sanidade (b): injetado × os mesmos negativos | **0,889** | ≥ 0,75 ✓ |
| **Principal: `var_fixa` × negativos** | **0,578** | faixa do meio (0,53–0,60) |
| com a logística | 0,582 | |
| com o GBM | 0,562 | |
| só defasagens (x·x₋ₖ, k=1..3) | **0,572** | |
| só valores ordenados (forma marginal) | 0,529 | |
| só momentos (média, \|x\|, x²) | 0,522 | |

Notas médias: `sobe` 0,725 · `desce` 0,731 · `var_fixa` 0,586 · negativos 0,556 · injetados 0,744.
Os negativos reais ficam acima do nulo sintético (0,556 contra 0,494): o trecho online se afasta um
pouco do fim do histórico mesmo sem quebra.

## Decisão

- **A sonda é válida** (as duas sanidades passam com folga) e as quebras "invisíveis" **não são
  totalmente invisíveis**. Há sinal fraco, e ele mora quase inteiro na **dependência de curto prazo**
  (defasagens 0,572), não em forma nem em escala. Isso confirma o excesso de ACF lag 1–2 do censo.
- Mas 0,578, com 400 pontos pós-quebra e um classificador por série, fica **abaixo** do que o Onyx já
  separa nesse mesmo grupo (TS-AUC 0,60 no subconjunto `var_fixa`, [ST1](../teto-offline/ST1-segundo-estagio.md)).
  Não há evidência de um eixo grande que o Onyx perca.
- **Pela regra registrada, o passo 3 sai** (não atinge 0,60). O que sobra dele é uma pista: dependência
  de curto prazo é o único eixo com excesso nas `var_fixa`, e as colunas de dependência do Onyx foram
  medidas como fracas e mal calibradas (HISTORICO §7).
- Leitura para o 0,68: o terço visível precisaria ser ordenado perto de 0,94. É o ramo
  "ordenar melhor o visível", e é para lá que apontam o passo 2 (professor/aluno) e o 4.
