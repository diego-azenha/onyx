"""C1(ii) -- o vetor `feature_contri` do LightGBM (B6 + conjunto penalizado estendido).

O teste central aqui é o de NO-OP: o B6 (`feature_contri_meta`) está ADOTADO e todas as medições da
campanha foram feitas com o código anterior, que penalizava só `meta_h0*`. Se a extensão do C1(ii)
mudasse o vetor quando ela está desligada, todo braço medido depois deixaria de ser comparável com os
anteriores -- a mesma classe de erro que produziu a retratação do D1.
"""
from dataclasses import replace

from sbrt.config import load_config
from sbrt.model.train import _feature_contri

COLS = [
    "meta_h0_scale", "meta_t", "conformal_logm_abs", "conformal_logm_abs_reset",
    "mmd_joint_slow", "mmd_joint_slow_cal", "cusum_var_up",
]


def _lgb():
    return load_config().lightgbm


def test_desligado_reproduz_o_codigo_anterior():
    """Com só o B6 ligado, o vetor é exatamente o que a implementação antiga produzia."""
    cfg = _lgb()
    anterior = [cfg.feature_contri_meta if c.startswith("meta_h0") else 1.0 for c in COLS]
    assert _feature_contri(cfg, COLS) == anterior


def test_nada_penalizado_devolve_none():
    """Sem B6 e sem C1(ii), nenhum `feature_contri` entra nos params do LightGBM."""
    cfg = replace(_lgb(), feature_contri_meta=1.0)
    assert _feature_contri(cfg, COLS) is None


def test_colunas_sem_multiplicador_sao_no_op():
    """Declarar colunas com `feature_contri_extra=1.0` não pode penalizar nada por acidente."""
    cfg = replace(_lgb(), feature_contri_meta=1.0,
                  feature_contri_extra_cols=("conformal_logm_abs",), feature_contri_extra=1.0)
    assert _feature_contri(cfg, COLS) is None


def test_extra_usa_multiplicador_proprio_e_nao_o_do_b6():
    cfg = replace(_lgb(), feature_contri_meta=0.6,
                  feature_contri_extra_cols=("conformal_logm_abs", "mmd_joint_slow_cal"),
                  feature_contri_extra=0.5)
    got = dict(zip(COLS, _feature_contri(cfg, COLS)))
    assert got["meta_h0_scale"] == 0.6
    assert got["conformal_logm_abs"] == 0.5
    assert got["mmd_joint_slow_cal"] == 0.5


def test_casamento_e_exato_nao_por_prefixo():
    """A razão de ser do casamento exato: `conformal_logm_abs_reset` está em `_MONO` (é acumulador de
    evidência, não rastreador de `t`), e `mmd_joint_slow` cru não é a coluna que o xs-SHAP acusou.
    Um casamento por prefixo penalizaria os dois e conflataria o braço."""
    cfg = replace(_lgb(), feature_contri_extra_cols=("conformal_logm_abs", "mmd_joint_slow_cal"),
                  feature_contri_extra=0.5)
    got = dict(zip(COLS, _feature_contri(cfg, COLS)))
    assert got["conformal_logm_abs_reset"] == 1.0
    assert got["mmd_joint_slow"] == 1.0
    assert got["cusum_var_up"] == 1.0
