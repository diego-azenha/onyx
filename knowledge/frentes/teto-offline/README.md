# Frente: teto offline e reconstrução do problema

**Aberta em:** 2026-09-30 · **Motivação:** o sprint de surpresa acumulada mostrou que a fraqueza do
Onyx não é velocidade. 57% do peso da métrica está em pares com ≥100 pontos pós-quebra, e ali o Onyx
separa só 0,65–0,70 ([diário 2026-09-29](../../diario/2026-09-29.md)). A pergunta: quanto estes dados
permitem separar quando a evidência já está toda lá, e que arquitetura explora isso em tempo real?

## Resumo executivo (madrugada de 2026-09-30)

**O teto não é de informação no sinal, é de amostra de EVENTOS de quebra, e o que paga é reduzir a
variância do aprendiz.**

1. **O Onyx já está no teto das features de duas amostras** ([T1](T1-teto-offline.md)): com τ
   conhecido, 118 features empatam com ele (0,666 em L=200). ROCKET, *variance ratios*, Chronos
   (modelo de fundação), excursões e segundo estágio não acrescentam nada.
2. **O Onyx é limitado por amostra** ([C1](C1-curva-aprendizado-onyx.md)): ×2 séries = +0,0144. E o
   limite é **só de positivos** ([C2](C2-positivos-ou-negativos.md)): negativos a mais rendem zero.
3. **Não existe fonte de eventos extras que sirva:** re-corte (duplica eventos, −0,012), transplante
   (caricatura paramétrica) e dados de 2025 (permitidos, mas de outra distribuição; AUC de domínio
   0,962).
4. **Reduzir variância funciona:** [E1](E1-extra-trees.md) (`extra_trees`) = **+0,0071
   [+0,0038; +0,0102]**, 4/4 sementes positivas, 0,6278 → **0,6349**. É o maior ganho isolado do
   projeto. Réplica na partição 43 e empilhamentos (V1–V6) na fila.
5. **Produção:** o refit com 100% das séries ([R1](R1-refit-completo.md)) deu resultado inconclusivo
   (o sinal troca entre sementes) e ficou desligado. O E1 está **adotado** (partições 42 e 43), e o
   notebook de submissão foi regenerado e verificado bit a bit.

| ID | Experimento | Status |
|---|---|---|
| T1 | [Teto offline com fronteira conhecida e prefixo contaminado](T1-teto-offline.md) | concluído: **teto baixo**, 0,666 com τ conhecido e L=200, igual ao Onyx |
| T2 | [Representação aprendida (ROCKET) no mesmo oráculo](T2-representacao-aprendida.md) | concluído: ROCKET não acrescenta (0,600); a curva de aprendizado do oráculo **não tem platô** |
| C1 | [O Onyx é limitado por amostra? (5k contra 10k séries)](C1-curva-aprendizado-onyx.md) | **concluído: sim, ×2 de dados = +0,0144** |
| A1 | [Aumento por re-corte de séries reais](A1-recorte.md) | **descartado: piora −0,012**; duplicar eventos aumenta a variância |
| C2 | [O limite está nos positivos ou nos negativos?](C2-positivos-ou-negativos.md) | **concluído: 100% positivos**; metade dos negativos deu até mais que tudo |
| A2 | [Eventos novos por transplante de assinatura](A2-transplante.md) | **descartado** offline: não soma em nenhum λ |
| D25 | [Dados da edição 2025](D25-dados-2025.md) | **permitidos** (o HISTORICO estava errado), mas **descartados**: não transferem e pioram (−0,019) |
| **E1** | [Extra-trees no Onyx](E1-extra-trees.md) | **ADOTADO**: part. 42 +0,0071 [+0,0038; +0,0102] (0,6349); part. 43 +0,0115 [+0,0080; +0,0159]; notebook verificado |
| P1 | [Poda para as top-65 features](P1-poda.md) | refeito sobre o E1 como V4 |
| ST1 | [Segundo estágio pequeno (Onyx + famílias)](ST1-segundo-estagio.md) | **negativo** (0,6267 contra 0,6278): a AUC por subconjunto não vira ordenação global |
| X1 | [Calibração por excursões + teste de deriva](X1-excursoes.md) | **negativos**: a calibração tira poder; o H0 é estacionário |
| T3 | [Embeddings de modelo de fundação (Chronos-Bolt)](T3-modelo-de-fundacao.md) | **negativo**: 0,53 sozinho, não soma |
| N1 | [Subamostrar negativos no treino](N1-subamostra-negativos.md) | refeito sobre o E1 como V2 |
| V1–V8 | [Empilhar redução de variância sobre o E1](V-empilhamento.md) | rodando (V1 nulo) |
| rep43 | Réplica do E1 na partição 43 | **replicado**: +0,0115, IC exclui 0 em todos os buckets |
| R1 | [Refit com 100% das séries (produção)](R1-refit-completo.md) | **inconclusivo** (+0,0041 / −0,0053 por semente); código pronto, desligado |
