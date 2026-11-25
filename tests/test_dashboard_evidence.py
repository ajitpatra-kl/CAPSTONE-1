from contextlib import nullcontext
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import app


@pytest.mark.parametrize(("fields", "expected"), [
    ({"vehicle_number": "KA01AB1234"}, "KA01AB1234"),
    ({"plate_candidate": "4SH4775"}, "4SH4775 (Unverified)"),
    ({"anpr_status": "Text Unreadable", "plate_crop_path": "missing.png"}, "Plate Detected — Text Unreadable"),
    ({"anpr_status": "Plate Not Detected"}, "Plate Not Detected"),
    ({"anpr_status": "Processing Failed", "plate_bbox": [0, 0, 4, 4]}, "Processing Failed"),
])
def test_plate_outcomes(fields, expected):
    assert app.violation_plate_outcome(fields) == expected


def test_clipped_display_fallback_leaves_original_unchanged(tmp_path):
    original = tmp_path / "event.png"
    bgr = np.full((10, 12, 3), [5, 10, 20], np.uint8)
    assert cv2.imwrite(str(original), bgr)
    contents = original.read_bytes()
    violation = {
        "vehicle_crop_path": str(tmp_path / "absent.png"), "original_frame_path": str(original),
        "bbox": [-3.4, 2.2, 20.4, 9.1], "frame_number": 52, "plate_frame_number": 60,
    }
    image, frame = app.violation_vehicle_image(violation)
    assert frame == 52
    np.testing.assert_array_equal(image, bgr[2:10, :, ::-1])
    assert original.read_bytes() == contents
    assert not (tmp_path / "absent.png").exists()


def test_evidence_card_uses_exact_track_and_source_frames(monkeypatch, tmp_path):
    ui = Mock()
    ui.expander.side_effect = lambda *args: nullcontext()
    ui.columns.side_effect = lambda count: [nullcontext() for _ in range(count)]
    monkeypatch.setattr(app, "st", ui)
    base = {"class_name": "car", "speed_kmh": 75, "speed_limit_kmh": 60, "excess_kmh": 15,
            "timestamp_sec": 2, "anpr_status": "Text Unreadable"}
    original = tmp_path / "event.png"
    plate = tmp_path / "plate.png"
    enhanced = tmp_path / "enhanced.png"
    vehicle = tmp_path / "vehicle.png"
    for image_path in (original, plate, enhanced, vehicle):
        assert cv2.imwrite(str(image_path), np.zeros((10, 12, 3), np.uint8))
    app.render_evidence_viewer([
        dict(base, track_id=12, frame_number=50, plate_frame_number=71, original_frame_path=str(original),
             bbox=[1, 1, 8, 8], plate_crop_path=str(plate), plate_enhanced_path=str(enhanced)),
        dict(base, track_id=26, frame_number=51, plate_frame_number=72, vehicle_crop_path=str(vehicle),
             plate_crop_path=str(tmp_path / "missing-plate.png")),
    ])
    assert [call.args[0].split(" · ")[0] for call in ui.expander.call_args_list] == ["Track #12", "Track #26"]
    assert [call.kwargs["caption"] for call in ui.image.call_args_list] == [
        "Vehicle crop · frame 50", "Native crop · frame 71", "Enhanced copy · frame 71", "Vehicle crop · frame 72",
    ]
    assert ui.image.call_args_list[1].args[0] == str(plate)
    assert ui.image.call_args_list[2].args[0] == str(enhanced)
    ui.info.assert_called_once_with("Plate crop image unavailable.")


def test_table_and_downloads_preserve_existing_reporting(monkeypatch, tmp_path):
    ui = Mock()
    ui.columns.side_effect = lambda count: [nullcontext() for _ in range(count)]
    ui.selectbox.return_value = 0
    monkeypatch.setattr(app, "st", ui)
    csv_path = tmp_path / "report.csv"
    tracks = tmp_path / "tracks.csv"
    evidence = tmp_path / "event.jpg"
    for path in (csv_path, tracks, evidence):
        path.write_bytes(path.name.encode())
    event = dict(track_id=1, class_name="car", speed_kmh=75, speed_limit_kmh=60,
                 excess_kmh=15, timestamp_sec=2, frame_number=50, evidence_path=str(evidence),
                 plate_candidate="4SH4775")
    app.render_violations_table([event])
    table = ui.dataframe.call_args.args[0]
    assert table.columns.tolist() == [
        "Track ID", "Vehicle", "Estimated Speed", "Speed Limit", "Excess Speed", "Timestamp", "Number Plate Status",
    ]
    assert table.iloc[0]["Number Plate Status"] == "4SH4775 (Unverified)"
    app.render_downloads(dict(csv_report_path=str(csv_path), tracks_csv_path=str(tracks), violations=[event]))
    assert [call.args[1] for call in ui.download_button.call_args_list] == [
        csv_path.read_bytes(), tracks.read_bytes(), evidence.read_bytes(),
    ]
