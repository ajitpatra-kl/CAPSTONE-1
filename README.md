# AI Traffic Violation Detection

## Number plate recognition

ANPR uses two independent pretrained models on CPU:

| Stage | Model and license | Input and output |
| --- | --- | --- |
| Plate detection | Subash's Indian plate YOLO checkpoint. The publisher identifies the detector as AGPL-3.0 and notes separate dataset terms. The model card labels the repository for non-commercial research and education. [Model card](https://huggingface.co/subash1652007/indian-plate-ocr) · [Public weights](https://huggingface.co/subash1652007/indian-plate-ocr/resolve/82232c6220919c8c60d148ad74eb13dbea37e260/yolo_best.pt) | Original BGR vehicle crop. Ultralytics resizes for inference and returns vehicle-crop-relative pixel boxes with detection scores. |
| Plate OCR | JaidedAI EasyOCR English generation 2 recognition model, distributed by the Apache-2.0 EasyOCR project. [Project](https://github.com/JaidedAI/EasyOCR) · [Public weights](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) | Grayscale plate crop. Returns raw text and an OCR confidence score. |

Install with `python -m pip install -r requirements.txt`. Save the detector as
`models/anpr/yolo_best.pt` and unzip `english_g2.pth` into `models/anpr/`.
Use the cited model sources. On Windows, `launch.bat` uses the project-local
`.venv` when present. To start manually, use `python -m streamlit run app.py`
with the environment that has EasyOCR installed.
The app reads ANPR paths and thresholds from `config/settings.json`. Missing weights
are logged and speed processing continues.

ANPR samples padded vehicle crops from original frames and keeps a bounded buffer per
track. When a violation is confirmed, it evaluates selected crops from that same ID
and favors a registration supported by multiple frames. Confidence in the CSV is the
OCR score. Crop filenames end in the source frame number because the clearest crop
may precede the violation frame. Small, blurred, obscured or unsupported plates
produce an empty registration. Validation covers the standard Indian state/RTO
format, and OCR is optimized for single-line plates.

The Streamlit dashboard detects and tracks vehicles in uploaded traffic video. Verified camera profiles enable metric speed estimates and overspeed evaluation. Red-light and lane violations are marked Coming Soon.

## Run locally

From the project directory, using Python 3.10 or newer and Git LFS for ForeSpeed acquisition:

```powershell
python -m pip install -r requirements.txt
python -X utf8 setup_dataset.py --download-model --download-dataset --verify
python -m streamlit run app.py
```

Open http://localhost:8501 and choose Vehicle Overspeed Detection. The local Roboflow demo is selected when no video is uploaded. You can upload an MP4, AVI, MOV, or MKV instead. The setup script verifies the pretrained model with a warm-up and requests missing ForeSpeed recordings through Git LFS. The application can also download YOLO11n on first use.

## Roboflow demonstration

Save the [official vehicles.mp4](https://media.roboflow.com/supervision/video-examples/vehicles.mp4) to `dataset/roboflow/vehicles.mp4`. The original clip decodes to 538 frames at 3840 × 2160 (16:9), with 25 FPS and a 21.53 s container duration. Its profile is selected only when both its SHA-256 and decoded dimensions match.

The [Roboflow notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-estimate-vehicle-speed-with-computer-vision.ipynb) maps `[(1252, 787), (2298, 803), (5039, 2159), (-550, 2159)]` in native pixels to `[(0, 0), (24, 0), (24, 249), (0, 249)]` in metres. The two off-frame vertices are intentional. No scaling or clipping is applied to the calibration points. Speeds use the existing ByteTrack IDs and bottom-centre anchors within the reference road polygon.

The notebook estimates longitudinal displacement from integer road coordinates over a nominal-FPS window. The application uses floating-point road coordinates, both axes, actual presentation timestamps, and its existing median smoothing. The notebook's road measurements are a reference for estimated speed. Physical accuracy has not been independently verified. Other videos use their matched profile or the existing approximate or unavailable speed modes.

## Audit

The initial suite collected 52 tests. It passed 50 and skipped two model tests because the YOLO11n weights were missing. The Streamlit dashboard, YOLO detection integration, mathematical homography, violation timing, and report generators were present. The T1/T2 filename rules selected approximate matrices, the speed estimator assigned an arbitrary metres-per-pixel scale to unknown videos, and the benchmark selected the fastest track as the reference vehicle. The tracker also called an incompatible ByteTrack API and quietly switched to IoU after an exception. Several paths could reuse stale observations or replace missing presentation timestamps with nominal FPS.

The pipeline now binds a camera profile to the original video bytes (SHA-256) and decoded resolution. YOLO applies cross-class suppression to overlapping vehicle detections. The tracker uses Ultralytics ByteTrack and reports its active algorithm. Only current observations contribute to speed and evidence. Camera projection uses the bottom centre of the detection box, and the estimator takes the median of road-plane velocity slopes over a recent window using the source presentation timestamps. Missing or non-monotonic timestamps and interrupted tracks break speed and violation continuity. Confirmed events require the configured duration and at least three valid above-limit observations. The silent H.264 playback preview uses the average measured rate when source timestamps are valid. Use the original video for timing-sensitive review.

Outputs go to `outputs/videos`, `outputs/evidence`, and `outputs/reports`. Confirmed events produce an annotated evidence image and a lossless PNG of the decoded frame. The dashboard offers the annotated video, vehicle CSV, and violation CSV as downloads. Missing speed values are blank in vehicle records and marked unavailable in summaries.

## ForeSpeed T1P1 calibration

The local ForeSpeed checkout has both original Git LFS recordings. Their SHA-256 values match the LFS object IDs and both videos decode to the full frame count.

| Recording | Camera | Decoded resolution | Frames | Source timing |
| --- | --- | ---: | ---: | --- |
| Eufy T1P1 | T8441 (CAM-D) | 2304 × 1296 | 988 | Variable frame rate, final PTS 65.802 s |
| Kasa T1P1 | KC105 (CAM-E) | 1920 × 1080 | 900 | Variable frame rate, final PTS 59.948 s |

Table 1 reports **29 mph (46.67 km/h)** for the T1P1 pass. The authors checked the speedometer against a laser sensor and cross-checked the speedometer against an independent GPS app during the pass. The benchmark provides one integer-mph reference for the pass. Table 2 identifies the cameras and export formats. [ForeSpeed paper](https://www.mdpi.com/2313-433X/12/10/488)

Figure 2 illustrates the views. **Figure 4(a)** shows a low-perspective road grid labelled **256 × 144 inches (6.5024 × 3.6576 m)**. Its grid is drawn on an illustration. Inspection of frames from both recordings showed distinct framing and visible road joints, but the source materials do not identify all four grid vertices as measurable physical points in either native video frame. They also do not provide native-frame correspondences, per-camera lens correction, or an independently measured set of check points. A small residual from fitting four points would only show agreement with those four inputs. [Figure 4](https://mdpi-res.com/jimaging/jimaging-12-00488/article_deploy/html/images/jimaging-12-00488-g004-550.jpg) · [Dataset documentation](https://github.com/ampedsoftware/forespeed)

Both profiles therefore have **calibration unavailable** status. Establishing a camera mapping requires the native-frame (u, v) positions of at least four identifiable coplanar road points paired with measured (X, Y) coordinates in metres, independent points or distances for verification, and confirmation of applicable lens correction and road-plane coverage. A future verified entry can supply its source, fitting points, held-out points, and tolerance in `config/dataset_config.json`. A benchmark comparison also requires identification of the actual pass vehicle's track.

## Validation

```powershell
python -m pytest tests/ -v
```

The synthetic pipeline test verifies a known 54 km/h trajectory and one confirmed event with two evidence images, an annotated video, and both CSVs. The dataset integration tests independently check the originals, identities, decoded frame counts and variable timestamps. Neither provides physical validation of the ForeSpeed camera mappings. The suite passed all 57 tests with one upstream deprecation warning on Python 3.12.

### Results on the original T1P1 recordings

| Recording | Processed frames | YOLO detections | ByteTrack IDs | Reference speed | Estimated speed | Absolute and percentage error |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Eufy CAM-D | 988 | 871 | 4 | 46.67 km/h | Unavailable | Unavailable |
| Kasa CAM-E | 900 | 923 | 3 | 46.67 km/h | Unavailable | Unavailable |

Both runs completed with zero invalid timestamps and exported fully decodable H.264 videos and two CSVs per recording. The Kasa run also passed Streamlit upload, processing, preview, download-control, and upload-clear checks. Neither run evaluated overspeed because the road mappings are unavailable.

Visual frame checks showed the pickup under Eufy ID 3 during the central pass and a two-frame Eufy ID 4 fragment at the right edge. The Kasa pickup retained ID 2 through the inspected pass frames. The pretrained detector sometimes labelled the pickup as a car. Track ID totals include other vehicles and brief fragments.

