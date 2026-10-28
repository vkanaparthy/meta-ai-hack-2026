from smokespotter.places import Place, compass, nearby


def test_compass_and_phrase():
    assert compass(0) == "N" and compass(44) == "NE" and compass(350) == "N" and compass(200) == "SSW"
    places = (Place("Julian", 33.0786, -116.6020, "town", 1500),
              Place("Ramona", 33.0417, -116.8681, "town", 20000),
              Place("Volcan Mountain", 33.1150, -116.6050, "peak", 0))
    out = nearby(33.12, -116.64, k=2, places=places)
    assert out[0]["name"] == "Julian" and out[0]["direction"] == "NW"
    assert out[-1]["kind"] == "peak"
    assert "km NW of Julian" in out[0]["phrase"]
