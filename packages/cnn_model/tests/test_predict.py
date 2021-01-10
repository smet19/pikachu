from cnn_model.predict import make_prediction


def test_make_prediction():
    result = make_prediction(
        img_url="https://upload.wikimedia.org/wikipedia/en/thumb/a/a6/Pok%C3%A9mon_Pikachu_art.png/220px-Pok%C3%A9mon_Pikachu_art.png",
        img_id="test")

    assert result["version"] == "0.1.0"
    assert result["prediction"] == "pikachu"
    assert (result["score"] > 0.5) and (result["score"] < 1.0)
    
