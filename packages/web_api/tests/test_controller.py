import json

from cnn_model import config as model_config
from cnn_model import __version__ as model_version

from web_api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == model_version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    test_url = "https://upload.wikimedia.org/wikipedia/en/thumb/a/a6/Pok%C3%A9mon_Pikachu_art.png/220px-Pok%C3%A9mon_Pikachu_art.png"
    test_id = "test"
    test_data = {'img_url': test_url, 'img_id': test_id}

    # When
    response = flask_test_client.post('/v1/predict',
                                      json=test_data)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['prediction']
    response_version = response_json['version']
    assert prediction == "pikachu"
    assert response_version == model_version