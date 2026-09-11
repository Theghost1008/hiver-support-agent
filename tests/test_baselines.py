from eval.baselines import trivial_baseline_fit, nn_baseline_fit, nn_baseline_predict

def test_trivial_baseline_picks_most_frequent():
    data = [
        {"text": "a", "intent":"x"},
        {"text": "b", "intent":"x"},
        {"text": "c", "intent":"y"},
    ]
    assert trivial_baseline_fit(data)=="x"

def test_nn_baseline_finds_semantic_match():
    data = [
        {"text":"My phone won't turn on", "intent":"hardware_issue"},
        {"text":"How do i turn on dark mode", "intent":"feature_howto"}
    ]
    embeddings, label = nn_baseline_fit(data)
    prediction = nn_baseline_predict("screen is black and unresponsive", embeddings,label)
    assert prediction == "hardware_issue"