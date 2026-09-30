# Premissas que o projeto nunca questionou

**Origem (2026-09-29).** Os competidores que mediram tetos entre 0,63 e 0,645 usam a mesma abordagem
que o Onyx. Enquanto isso, alguém saiu de 0,652 para 0,68 em três semanas, e um salto desse tamanho
tão tarde costuma vir de uma descoberta, não de polimento. A pergunta que organiza esta pasta: *o que
precisa ser verdade no problema para 0,68 existir, e o que nunca questionamos?*

Cada premissa tem uma ficha com: a afirmação, o que o repositório já sabe sobre ela (com link), o
status e os experimentos que a testam.

| # | Premissa | Status | Prioridade |
|---|---|---|---|
| 1 | [A quebra aparece em resumos da série](01-surpresa-acumulada.md) | **testada, nula**: a acumulação detecta mais rápido, mas lê os mesmos eixos que o Onyx ([frente](../frentes/surpresa-acumulada/README.md)) | fechada |
| 2 | [Cada série é julgada sozinha](02-estado-entre-series.md) | aberta; **contradiz** o HISTORICO §11 | 2ª |
| 3 | [O modelo aprende probabilidades, mas o placar premia a ordem](03-objetivo-de-ordenacao.md) | testada uma vez (R3), numa régua antiga; o código está quebrado | reauditoria |
| 4 | [Os dados são séries financeiras genéricas](04-gerador-das-quebras.md) | **achado:** ~⅔ das quebras são indistinguíveis dos negativos em 10 eixos simples | **candidata a próxima** |

**Por que 1 e 2 primeiro:** as duas mexem em algo que os competidores com teto de 0,63 não fizeram, e
se combinam. A surpresa acumulada produz um número que já tem o mesmo significado em qualquer série,
o que resolve boa parte da comparação entre séries.
