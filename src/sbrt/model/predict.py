"""ModelEnsemble.predict_one (plano §8.4)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from sbrt.features.assembly import to_array


@dataclass
class ModelEnsemble:
    boosters: list
    feature_order: tuple
    predict_num_threads: int = 1
    fold_evals: list = field(default_factory=list)  # diagnóstico (plano §9.1) — persistido em fold_evals.json
    base_rate_curve: dict | None = None  # plano_acao_v1_para_v2.md A2 — metadado de treino, NÃO usado
    # em predict_one (ver docstring): somar de volta é neutro para TS-AUC por invariância C1, mas
    # infla o score em cenários sintéticos fora da distribuição real. Fica salvo para diagnóstico
    # (ex.: reconstruir o resíduo de treino) e para quem quiser reabilitar a calibração absoluta.

    @classmethod
    def load(cls, path: str | Path) -> "ModelEnsemble":
        path = Path(path)
        boosters = joblib.load(path / "boosters.joblib")
        feature_order = tuple(json.loads((path / "feature_schema.json").read_text(encoding="utf-8")))
        meta = json.loads((path / "ensemble_meta.json").read_text(encoding="utf-8"))
        fold_evals_path = path / "fold_evals.json"
        fold_evals = json.loads(fold_evals_path.read_text(encoding="utf-8")) if fold_evals_path.exists() else []
        base_rate_path = path / "base_rate_curve.json"
        base_rate_curve = json.loads(base_rate_path.read_text(encoding="utf-8")) if base_rate_path.exists() else None
        return cls(
            boosters=boosters,
            feature_order=feature_order,
            predict_num_threads=meta["predict_num_threads"],
            fold_evals=fold_evals,
            base_rate_curve=base_rate_curve,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.boosters, path / "boosters.joblib")
        (path / "feature_schema.json").write_text(json.dumps(list(self.feature_order), indent=2), encoding="utf-8")
        (path / "ensemble_meta.json").write_text(
            json.dumps({"predict_num_threads": self.predict_num_threads}), encoding="utf-8"
        )
        (path / "fold_evals.json").write_text(json.dumps(self.fold_evals, indent=2), encoding="utf-8")
        if self.base_rate_curve is not None:
            (path / "base_rate_curve.json").write_text(json.dumps(self.base_rate_curve), encoding="utf-8")

    def predict_one(self, feats: dict) -> float:
        """Média dos folds; num_threads=1; ordem de colunas fixada pelo schema salvo. SEM tqdm —
        é caminho de inferência real (plano §8, regra tqdm).

        plano_acao_v1_para_v2.md A2/A5: os boosters são treinados com `init_score = logit(p_hat(t))`
        (model/train.py), então `predict()` (sem `raw_score`) já devolve `sigmoid(raw)` -- o resíduo
        transversal, SEM o offset de taxa-base (LightGBM nunca readiciona `init_score` para dados
        novos). Deliberadamente NÃO somamos o offset de volta aqui: por invariância C1 (plano técnico
        §1.2) isso é neutro para a TS-AUC oficial (desloca todas as séries vivas igualmente em cada
        t), mas somá-lo de volta infla o score em cenários fora da distribuição de treino (medido:
        piorou a suíte de robustez em T6/T9/T10/T12/T12b — decisão tomada com o usuário após ver o
        efeito). O score aqui é o resíduo puro; não é uma probabilidade calibrada absoluta."""
        x = to_array(feats, self.feature_order).reshape(1, -1)
        preds = [b.predict(x, num_threads=self.predict_num_threads)[0] for b in self.boosters]
        return float(np.mean(preds))
