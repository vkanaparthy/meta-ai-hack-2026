from smokespotter.muse import parse_verdict
from smokespotter.sam import parse_detections

# Sample from meta-sam-parser's README: one frame, two objects, 320x334 video.
SAMPLE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)


def test_parse_detections_scales_to_original_pixels():
    dets = parse_detections(SAMPLE, scale=2.0)
    assert [(d.frame, d.object_id) for d in dets] == [(0, "0"), (0, "1")]
    d0 = dets[0]
    assert (d0.x0, d0.y0, d0.x1, d0.y1) == (422, 456, 542, 510)  # right/bottom are exclusive
    assert 422 <= d0.base_x <= 542
    assert d0.area_px > 0


def test_empty_output_is_no_detections():
    assert parse_detections("") == []


def test_parse_verdict():
    v = parse_verdict('Sure.\n{"is_smoke": true, "confidence": 0.82, "reason": "rising plume"}')
    assert v.is_smoke and v.confidence == 0.82
    assert not parse_verdict("no json here").is_smoke


def test_verdict_retries_empty_replies(monkeypatch):
    """A reasoning model can return an empty reply; ask again instead of reading it as 'not smoke'."""
    from smokespotter import muse

    replies = iter(["", '{"is_smoke": true, "confidence": 0.9, "reason": "rising column"}'])
    monkeypatch.setattr(muse, "ask_vision", lambda *a, **k: next(replies))
    v = muse._ask_verdict("prompt", [], "verify")
    assert v.is_smoke and v.reason == "rising column"
