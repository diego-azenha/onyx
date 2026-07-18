from sbrt.config import Config, load_config


def test_load_config_returns_typed_config(cfg):
    assert isinstance(cfg, Config)
    assert cfg.h0.ar_order == 10
    assert cfg.bayes.hazards == [0.02, 0.01, 0.0025]
    assert set(cfg.gates.scenarios.keys()) == {
        "t1", "t2", "t3", "t4", "t5", "t5b", "t6", "t7", "t8",
        "t9", "t10", "t11", "t12", "t12b", "t13",
    }


def test_load_config_is_reloadable():
    cfg1 = load_config()
    cfg2 = load_config()
    assert cfg1.seed == cfg2.seed
    assert cfg1.h0.ar_order == cfg2.h0.ar_order
