from backend.recommender import load_artifacts, recommend_foods


def test_load_artifacts_shapes_match():
    data, cosine_sim = load_artifacts()
    assert len(data) > 0
    assert cosine_sim.shape[0] == cosine_sim.shape[1]
    assert cosine_sim.shape[0] == len(data)


def test_recommend_foods_returns_ranked_items():
    data, cosine_sim = load_artifacts()
    results = recommend_foods("chicken", data, cosine_sim, top_n=5)

    assert results
    assert len(results) <= 5

    first = results[0]
    expected_keys = {
        "name",
        "cuisine",
        "food_tag",
        "type",
        "restaurants",
        "restaurant_source",
        "score",
    }
    assert expected_keys.issubset(first.keys())


def test_recommend_foods_handles_unknown_query():
    data, cosine_sim = load_artifacts()
    results = recommend_foods("totally_unknown_food_token_xyz", data, cosine_sim)
    assert results == []
