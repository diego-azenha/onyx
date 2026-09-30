# A2 — Eventos de quebra novos por transplante de assinatura

**Frente:** [teto offline](README.md) · **Status:** **descartado** no teste offline (antes de treinar o
Onyx) · **Data:** 2026-09-30

## Ideia

Criar eventos de quebra **novos** (o re-corte do [A1](A1-recorte.md) só cria vistas novas dos
4.967 reais). Da quebra real j (≥100 pontos pós-quebra) extrai-se a assinatura: Δφ de um AR(5)
pré/pós, a razão de escala, o mapa de quantis dos resíduos padronizados pré → pós e a Δmédia. Ela é
aplicada à continuação **real** de outra série (a hospedeira), a partir de um τ' sorteado. Ficam
~5k quebras × 10k hospedeiras de combinações. `scripts/a2_transplante.py`.

**Vazamento evitado:** a doadora é sorteada entre as séries do **mesmo fold** da hospedeira. Uma
primeira versão por paridade de id deixaria a assinatura de uma série validada entrar no treino
de outro fold.

## Teste barato de realismo (oráculo com L=200, features do T1)

| Teste | λ = 1 (assinatura inteira) | λ = 0,5 | λ = 0,3 |
|---|---|---|---|
| Treino só sintético → teste real | 0,637 | 0,621 | 0,594 |
| Treino real → teste sintético | 0,736 | 0,623 | 0,562 |
| CV no real: só real / **real + sintético** | 0,6665 / **0,6656** | 0,6665 / 0,6653 | 0,6665 / 0,6602 |

λ encolhe a assinatura. A estimada carrega ruído de estimação que exagera a mudança: com λ=1, as
sintéticas são mais fáceis que as reais (0,736).

## Decisão

**Descartado.** Os eventos transplantados transferem parcialmente (0,637), mas **somados aos reais não
melhoram nada** em nenhum λ. É uma caricatura paramétrica: reproduz escala, AR e forma marginal,
e o que torna as quebras reais difíceis não está nisso. Não vale gastar treino do Onyx com ele.
