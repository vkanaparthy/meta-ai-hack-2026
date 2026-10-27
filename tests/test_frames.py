from smokespotter.frames import load_sequence, parse_frame_name


def test_parse_name():
    assert parse_frame_name("1465065268_-02400.jpg") == (1465065268, -2400)
    assert parse_frame_name("1465067728_+00060.jpg") == (1465067728, 60)
    assert parse_frame_name("index.html") is None


def test_load_sequence_orders_and_labels(tmp_path):
    for n in ["1465065328_-00060.jpg", "1465065268_-00120.jpg", "1465065388_+00000.jpg", "x.mp4"]:
        (tmp_path / n).write_bytes(b"")
    seq = load_sequence(tmp_path)
    assert [f.offset for f in seq] == [-120, -60, 0]
    assert [f.has_smoke for f in seq] == [False, False, True]
