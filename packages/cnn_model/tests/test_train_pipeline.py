from cnn_model.train_pipeline import run_eval


def test_run_eval_return_good_score():
    results = run_eval()

    assert results is not None
    assert (results[1] >= 0.7) and (results[1] < 1.0)