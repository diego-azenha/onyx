# Base de conhecimento do Onyx

Tudo o que o projeto sabe fica aqui: como o modelo funciona, o que já foi tentado, o que cada mudança
rendeu e o que está em andamento. Substitui a antiga pasta `docs/` (migrada em 2026-09-29, ver
[diário](diario/2026-09-29.md)).

## Mapa

| Pasta | O que tem | Comece por |
|---|---|---|
| [`premissas/`](premissas/README.md) | As premissas que o projeto nunca questionou, cada uma com status e evidência | [README](premissas/README.md) |
| [`frentes/`](frentes/) | Frentes de trabalho ativas, cada uma com desenho + uma ficha por experimento | [surpresa acumulada](frentes/surpresa-acumulada/README.md) |
| [`diario/`](diario/) | Uma entrada por dia: o que foi feito, com links para fichas e commits | a entrada mais recente |
| [`modelo/`](modelo/MODELO.md) | Como o Onyx funciona: formulação, decisões, fórmulas (`§N`) | [MODELO.md](modelo/MODELO.md) |
| [`operacao/`](operacao/NOTAS_AGENTES.md) | Invariantes, contratos, comandos, **protocolo de medição**, pegadinhas | [NOTAS_AGENTES.md](operacao/NOTAS_AGENTES.md) §1, §2, §5 |
| [`historico/`](historico/HISTORICO.md) | Toda rodada passada com resultado medido e decisão; backlog; plano da semana de 27/07 | [HISTORICO.md](historico/HISTORICO.md) §1 (placar) |
| [`campanhas/`](campanhas/) | Relatórios completos das campanhas (EXP0, X0–X4, B/C, polimento) | — |
| [`ideias/`](ideias/) | Brainstorms e análises que ainda não viraram experimento | — |
| [`_modelos/`](_modelos/experimento.md) | Template de ficha de experimento | — |

## Convenções

1. **Uma ficha por experimento**, em `frentes/<frente>/<ID>-<nome>.md`, copiada de
   [`_modelos/experimento.md`](_modelos/experimento.md). A seção **Hipótese** é escrita e salva
   **antes** de rodar qualquer medição, seguindo o protocolo de [NOTAS_AGENTES §5](operacao/NOTAS_AGENTES.md).
   Resultado e decisão entram depois, sem editar a hipótese.
2. **Uma entrada de diário por dia de trabalho**, em `diario/AAAA-MM-DD.md`, com links para as fichas
   tocadas. O diário diz *o que aconteceu*; a ficha diz *o que o experimento mostrou*.
3. **Números sempre com a régua**: grade (`full` = board, `thin` = antiga), partição de folds
   (42/43), K sementes e baseline. Um número sem régua não é comparável a nada
   ([HISTORICO §15.1](historico/HISTORICO.md)).
4. **Links relativos** entre arquivos. Ao citar uma seção, use `ARQUIVO.md §N`.
5. Os arquivos migrados de `docs/` foram movidos **inteiros**, então toda referência `§N` antiga
   continua resolvendo.

## Incumbente de referência (atualizar a cada adoção)

| O quê | Valor | Onde |
|---|---|---|
| Pacote atual | receita B6+C6 + X1 soft + D1 (`linear_tree`) + B3 (v-EMA) + **E1 (`extra_trees`)** | [E1](frentes/teto-offline/E1-extra-trees.md) |
| TS-AUC OOF, grade do board, partição 42, K=4 | **0,6349** (`oof_e1_vema_bag4`); sem E1, 0,6278 (`oof_b0_vema_bag4`, reconstruído neste clone) | [B0](frentes/surpresa-acumulada/B0-incumbente.md) |
| Partição 43, K=2 | 0,6308 (E1) contra 0,6193 (B0) | [E1](frentes/teto-offline/E1-extra-trees.md) |
| Placar oficial conhecido | V4 `5e42ff5` = 0,6201; B6+C6 ≈ 0,6267. O E1 ainda **não foi submetido** | [NOTAS §5](operacao/NOTAS_AGENTES.md) |
| Teto legítimo e alvo realista | ~0,645–0,65 no placar; o topo (68,71) coincide com E1 + comprimento do online conhecido, que não é previsível de forma legítima. E1 estimado por volta do rank 85 de 797. **A fase de submissão fecha em 03/10, 16:00** | [L1](frentes/teto-offline/L1-comprimento-e-o-topo.md) |
| Notebook de submissão | regenerado com o E1 e verificado bit a bit em 2026-09-30 | `submission_notebook.ipynb` |

## Redirecionamento: caminhos antigos → onde estão agora

O código, o YAML e os notebooks citam os caminhos antigos em ~90 arquivos. Eles **não** foram
reescritos em massa, pela mesma política da [NOTAS §10](operacao/NOTAS_AGENTES.md): as docstrings de
`src/` entram no `submission_notebook.ipynb`, que é verificado bit a bit. Ao editar um arquivo por
outro motivo, atualize a referência de passagem. Use esta tabela para resolver qualquer citação.

| Citado no código como | Está em |
|---|---|
| `docs/MODELO.md`, `docs/PLANO_TECNICO.md`, "plano §N", "plano técnico" | [`modelo/MODELO.md`](modelo/MODELO.md) (mesma numeração `§N`) |
| `docs/ARTIGO.md` | [`modelo/ARTIGO.md`](modelo/ARTIGO.md) |
| `docs/DIAGNOSTICO_ESTRUTURAL.md` | [`modelo/DIAGNOSTICO_ESTRUTURAL.md`](modelo/DIAGNOSTICO_ESTRUTURAL.md) |
| `docs/NOTAS_AGENTES.md`, `docs/CONTRACTS.md` (→ §2), `docs/PLANO_REPOSITORIO.md` (→ §1–§4 e MODELO §15) | [`operacao/NOTAS_AGENTES.md`](operacao/NOTAS_AGENTES.md) |
| `docs/HISTORICO.md` | [`historico/HISTORICO.md`](historico/HISTORICO.md) |
| `docs/DIAGNOSTICO_TS_AUC.md` | [`historico/HISTORICO.md`](historico/HISTORICO.md) §2 |
| `docs/PARECER_AUDITORIA_ONYX.md` (D1–D4, R0–R6) | [`historico/HISTORICO.md`](historico/HISTORICO.md) §3 |
| `docs/RESULTADOS_ROADMAP_R0_R6.md` | [`historico/HISTORICO.md`](historico/HISTORICO.md) §4 |
| `docs/PROPOSTA_FEATURES_V2.md` (F1–F6), `docs/RESULTADOS_FEATURES_V2.md` | [`historico/HISTORICO.md`](historico/HISTORICO.md) §5–§6 |
| `docs/INVESTIGACAO_FALHAS_V3.md` (P1–P4) | [`historico/HISTORICO.md`](historico/HISTORICO.md) §7 |
| `docs/RESULTADOS_P1_P4.md` (V4) | [`historico/HISTORICO.md`](historico/HISTORICO.md) §8 |
| `docs/BACKLOG_TSAUC.md` | [`historico/BACKLOG_TSAUC.md`](historico/BACKLOG_TSAUC.md) |
| `docs/PROXIMA_SEMANA.md` | [`historico/PROXIMA_SEMANA.md`](historico/PROXIMA_SEMANA.md) |
| `CAMPANHA_POLIMENTO.md` (raiz) | [`campanhas/CAMPANHA_POLIMENTO.md`](campanhas/CAMPANHA_POLIMENTO.md) |
| `docs/RELATORIO_EXP0.md` | [`campanhas/RELATORIO_EXP0.md`](campanhas/RELATORIO_EXP0.md) |
| `docs/RELATORIO_CAMPANHA_X0_X4.md` | [`campanhas/RELATORIO_CAMPANHA_X0_X4.md`](campanhas/RELATORIO_CAMPANHA_X0_X4.md) |
| `docs/RELATORIO_CAMPANHA_B_C.md` | [`campanhas/RELATORIO_CAMPANHA_B_C.md`](campanhas/RELATORIO_CAMPANHA_B_C.md) |
| `docs/RELATORIO_POLIMENTO.md` | [`campanhas/RELATORIO_POLIMENTO.md`](campanhas/RELATORIO_POLIMENTO.md) |
| `docs/BRAINSTORM_RUPTURA_V2.md` | [`ideias/BRAINSTORM_RUPTURA_V2.md`](ideias/BRAINSTORM_RUPTURA_V2.md) |
| `informacao_nao_capturada.md` (raiz) | [`ideias/informacao_nao_capturada.md`](ideias/informacao_nao_capturada.md) |
| `BRAINSTORM_RUPTURA_TSAUC.md`, `plano_acao_tsauc_consolidado.md` | **nunca versionados** neste repositório; o conteúdo usado está resumido em [HISTORICO §13](historico/HISTORICO.md) e [RELATORIO_CAMPANHA_X0_X4](campanhas/RELATORIO_CAMPANHA_X0_X4.md) |
