import pytest


@pytest.mark.xfail(reason="Training pipeline skeleton not implemented yet")
def test_pipeline_interfaces_exist():
    """
    Skeleton test: verifies that core functions/classes exist to be implemented.

    TODO for student:
    - Import functions from experiments.train.data and implement them
    - Implement LMFixedLengthCollator.__call__
    - Implement trainer.combine_loss/train/evaluate
    - Replace this xfail with real integration checks
    """
    from experiments.train import data, collate, trainer, config as cfg

    assert hasattr(data, "get_tokenizer")
    assert hasattr(data, "load_wikitext2_splits")
    assert hasattr(data, "to_torch_dataset")
    assert hasattr(collate, "LMFixedLengthCollator")
    assert hasattr(trainer, "combine_loss")
    assert hasattr(trainer, "train")
    assert hasattr(cfg, "TrainingConfig")
