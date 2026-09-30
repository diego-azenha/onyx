# L1: o valor de conhecer o comprimento do online, e o que isso diz sobre o topo do placar

**Data:** 2026-09-30 · **Tipo:** diagnóstico (NUNCA para produção) · **Script:** `scripts/l1_comprimento_diag.py`

## Por que medir

O topo do placar saltou de 0,652 para 0,687 nas últimas ~3 semanas
([premissa 4](../../premissas/04-gerador-das-quebras.md)). Dois participantes, no fórum
(["Where does the remaining headroom live?"](https://forum.crunchdao.com/t/where-does-the-remaining-headroom-live-sharing-measured-ceilings-from-a-whitening-based-approach/1206),
4/set), mediram tetos independentes do nosso e chegaram ao mesmo lugar:

| Participante | Representação | Teto medido (curva de aprendizado em nº de séries) |
|---|---|---|
| weak-van-weyenbergh | branqueamento de Rosenblatt, 119 features, ensemble | ~0,633 no placar ("100× os dados compram ~+0,006") |
| agreed-harshveer | 1.728 features, LightGBM ×12 | ~0,64 (AUC ≈ 65,6 − 206/√n) |
| **nós** | Onyx E1, 194 features | OOF 0,635 → placar estimado ~0,645 (âncora +0,010, [NOTAS §5](../../operacao/NOTAS_AGENTES.md)) |

O primeiro também mediu: um **oráculo que conhece τ** bate o modelo dele por só +0,019, e 82% dessa
vantagem é o preço estrutural de não conhecer τ. Isso é o mesmo que o nosso [T1](T1-teto-offline.md) mostrou.
Nenhuma modelagem do conteúdo das séries explica um salto de +0,035 acima desses tetos.

O vazamento fechado em 8/jun ([fórum](https://forum.crunchdao.com/t/leaderboard-comparability-after-the-june-8-real-time-data-access-fix-were-pre-fix-scores-rescored/1188))
deixava descobrir **o comprimento da série** durante o `infer`. Como τ é uniforme dentro do online, o
comprimento T dá o prior P(τ ≤ t) = t/T. A pergunta do L1: quanto isso vale somado a um modelo como o E1?

## Resultado (OOF do E1, partição 42, K=4, grade do board)

| Score | TS-AUC |
|---|---|
| E1 | 0,6349 |
| só o prior t/T (nenhuma informação do conteúdo) | **0,6312** |
| logit(E1) + logit((t+1)/(T+1)), soma simples | **0,6869** |
| logística cross-fit [E1, prior, log t, log T] | 0,6971 |

**O topo do placar (68,71) coincide com "um modelo do nosso nível + o comprimento do online".**

## T é previsível de forma legítima? Não.

- Comprimentos no treino: histórico ~U[1000, 5000], online ~U[10, 999], **correlação 0,005**
  (Spearman 0,005). O total hist+online tem 3.995 valores distintos em 10.000 séries.
- O participante do fórum testou 23 descritores para prever o comprimento do online: R² fora da amostra
  negativo.
- A premissa 4 já mostrava que o comprimento do histórico não carrega o rótulo (AUC 0,495).

## Leitura e decisão

1. **O salto do topo não é alcançável modelando o conteúdo das séries**, ao menos com tudo o que nós e
   dois outros participantes medimos. É compatível com acesso ao comprimento (ou a informação
   equivalente) durante o `infer`. Isso é circunstancial: **não é prova** de que alguém use um
   vazamento, e pode haver informação legítima que ninguém mediu.
2. **Não vamos procurar vazamentos.** Seria violar as regras (a organização invalida e desqualifica) e
   não é o objetivo do projeto.
3. **O alvo realista muda:** o teto legítimo medido fica em ~0,64–0,65 no placar. O corte do top-50 é
   0,650. O E1 (placar estimado ~0,645) já está perto desse teto. O maior ganho disponível hoje é
   **submeter o E1** (nunca submetido) e confirmar o nível no placar.
4. A avaliação final roda sob o protocolo corrigido e **sobre dados novos**. Quem depende de informação
   que não existe no conteúdo pode não manter a posição.

## Por que ninguém tinha visto

Todos os diagnósticos do projeto (T1, C1, C2, S6) mediram o **conteúdo** das séries. O prior t/T sozinho
já iguala o E1 (0,631 contra 0,635), e é ortogonal a ele. É a única peça de informação que conhecemos
com esse tamanho.

## O placar, relido pela API (16:45)

`api.hub.crunchdao.com/v2/competitions/structural-break-real-time/leaderboards/@default`, crunch 22,
release 234 (a mesma durante toda a fase de submissão, então todos são pontuados no mesmo teste público):

- **797 posições** (a leitura anterior, "151", pegou só uma página), ordenadas pela melhor submissão.
- rank 1: 68,71 · 3: 68,02 · 10: 66,9 · 25: 66,0 · 50: 65,0 · 100: 63,95 · 200: 62,7 · 300: 61,4 ·
  500: 57,6 · último: 45,1. Só **3** projetos passam de 68 e **9** de 67.
- A escala é normal: o V4 (0,6201) ficaria por volta do rank 250, e o E1 (~0,645 estimado) por volta do
  **rank 85, top 11%**.
- **Prazo:** a fase de submissão (`phase.end`) termina em **2026-10-03 16:00**, e não em 1º/10. A rodada
  vai até 31/10 (avaliação fora da amostra).

O salto acima de ~0,66 fica concentrado em poucos projetos. Isso é compatível com a leitura acima:
o grosso do pelotão está no teto que três representações independentes mediram.
