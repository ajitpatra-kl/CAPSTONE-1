# AI Traffic Violation Detection

## Unified traffic violation detection (System A)

On Windows, run `launch.bat`, or start Streamlit with
`python -m streamlit run app.py` from the project environment. The default
**Unified Monitoring** page accepts one upload and one or more violation
types. Use the scene preview to configure the selected types, then select
**Start processing**. Open **Violation Records → Unified sessions** to revisit
completed runs and export confirmed events, all candidates, track observations,
or the experiment summary. The earlier individual workflows are available from
the sidebar under **Standalone tools**. **Overview** and **Analytics** report
stored unified sessions, and **System & Settings** shows local model-file
checks and saved scene profiles. Within a run, filter confirmed events by
type, status, track, registration, or video time. Select an event to review
its exact evidence frame, optionally cue video playback, and download its
evidence ZIP.

## Temporal-Contextual Violation Verification (System B)

After **Unified Monitoring** finishes, select **Run TCVV Verification** in the
results. The same control is available under **Violation Records → Unified
sessions** for eligible saved sessions. Use the **TCVV run** selector to reopen
a result. Each explicit run gets a separate ID, timestamp, policy version and
configuration snapshot. The results view links to the original evidence and
shows the observations and reasons behind each decision.

TCVV starts with the same candidate IDs, tracks, timestamps and configuration
that System A saved. It reads `tracks.csv` and the session’s compact
`tcvv-context-v1.jsonl` sidecar. The sidecar adds timestamped raw and confirmed
signal observations, signal quality and onset, and per-frame lane overlap
flags. Stop-line geometry, crossing brackets, lane geometry, the speed limit,
calibration status, and evidence paths already come from System A. SQLite
stores System B in `tcvv_runs` and `tcvv_results`. Verification uses neither
video decoding nor inference. Historical sessions with only overspeed
candidates can run when their track export is available. Sessions with
red-light or lane-change candidates need the sidecar captured during unified
processing.

| Policy | Verification criteria |
| --- | --- |
| Overspeed | Consecutive, consistent speed estimates from verified calibration must exceed the configured limit for the required duration. Reference and assumed calibrations yield uncertain physical-speed decisions. |
| Red light | A continuous trajectory must cross the configured finite stop-line segment in the monitored direction, with observations on both sides and a stable red phase whose onset was observed before the crossing. |
| Lane change | A continuous, non-overlapping anchor trajectory must cross an adjacent boundary, establish travel direction, and stabilize on the destination side. The configured boundary classification and monitoring direction determine whether the transition is restricted. |

An uncertain result records an evidence gap or reliability issue. A rejection
requires contradicting observations, such as a stable non-red signal at a
crossing or an observed permitted transition. Signal quality is the existing
classifier’s heuristic margin, and speed estimates retain the limits of the
original measurement and calibration.

### Compare against labels

Open **System A vs System B** in the TCVV section. Comparison groups overlapping
overspeed candidate intervals for the same track and groups repeated crossing
submissions with the same track, violation type and evidence-frame bracket.
An episode is confirmed for a system if any of its candidates is confirmed.
Agreement counts episodes confirmed by both systems and episodes that System A
did not confirm and System B rejected. The tab also reports TCVV elapsed time
through result insertion and the ratio to measured System A processing time.

Upload a UTF-8 CSV with columns
`session_id,video_id,track_id,violation_type,timestamp,label`, using timestamps
in seconds and `overspeed`, `red_light`, or `lane_change` as the type. Label each
physical event once with `1` for a violation or `0` for a verified negative.
Include eligible missed violations and representative negatives. Use the
original candidate’s timestamp convention, which is the sustained-exceedance
decision time for overspeed and the crossing time for red light and lane change.
For an event without a candidate, annotate the equivalent video time and use
its System A track ID if available. Use a distinct unused track ID for each
untracked event.

Matching uses the selected session and video, the same track and violation
type, and the closest candidate timestamp in each episode within the chosen
tolerance (default 0.5 seconds). It maximizes one-to-one matches using distance
and deterministic ordering to prioritize eligible pairs. Unmatched positive
annotations count as misses. Unmatched predictions are reported as unscored.
Precision, recall, F1, false-positive and false-negative rates use the labeled
population. An uncertain outcome counts as a non-confirmation for these
metrics, and uncertain-event rates use all compared episodes. Label enough
system-only predictions and negative events for meaningful precision and
false-positive rates. Blank rates have a zero denominator.

### Repository audit and structure

The original Streamlit entry point is `app.py`, with `launch.bat` as the Windows
launcher and dependencies in `requirements.txt`. The three original processors
each owned a decode loop. They already shared the YOLOv11 vehicle detector and
Ultralytics ByteTrack wrapper in `modules/speed_detection`. Speed used
checksum-matched homographies, source timestamps, smoothing, and a sustained
exceedance rule. RLVD used HSV signal classification, observed lamp onset,
and a directed finite stop-line crossing. LCVD used native-frame piecewise
boundaries, stable lane transitions, and gap/overlap checks. Existing evidence
and reports were filesystem artifacts, with no shared database. ANPR supplied
plate detection, crop buffering, EasyOCR, and a dedicated Roboflow OCR comparison
workflow. Tests covered the independent rules, tracking, calibration, ANPR,
evidence, and the ForeSpeed inputs.

```text
app.py                         Streamlit entry point
launch.bat / requirements.txt  Launcher and dependencies
config/                        Application settings and calibration profiles
configs/{rlvd,lcvd}/            Existing saved scene geometries
modules/
  unified/
    pipeline.py                Shared input, inference, tracking and evidence
    engines.py                 Adapters to the three existing rule engines
    schema.py                  Observations, context, candidates and verification
    storage.py                 SQLite sessions, candidates, events and exports
    dashboard.py               Unified setup, video results and saved reports
  tcvv/                        Stored context, independent rules, run history and comparison
  speed_detection/
    speed_session.py            Shared calibration, timing and speed continuity
    vehicle_detector.py         Existing YOLOv11 wrapper
    vehicle_tracker.py          Existing per-video ByteTrack wrapper
  rlvd/ and lcvd/              Existing rules, geometry, editors and workflows
  anpr/                        Plate detection and OCR
  dashboard/                   Navigation, operations, analytics, and legacy reports
tests/test_unified.py           System A integration and Streamlit coverage
tests/test_tcvv.py              TCVV policy, storage, evaluation and Streamlit coverage
outputs/unified/                Local SQLite database and session artifacts
```

### From video to confirmed event

One OpenCV capture decodes the uploaded source during processing. Each frame
runs through one YOLOv11 detector and one session-local ByteTrack instance.
Selected engines receive the same track IDs and vehicle observations. IDs can
repeat across videos, so use `(session_id, track_id)` as the track key and a
SHA-256 video ID to identify source content. The UI scene preview runs before
the processing session.

The context keeps the latest 60 observations per track alongside the tracker's
state. Observations include frame number, timestamp, timestamp validity, class,
confidence, box, bottom-center point, track age and gap length. Speed, lane,
signal and stop-line distance are populated when the corresponding engine is
enabled. The per-frame CSV retains observations beyond the in-memory window.

The overspeed adapter shares calibration and continuity logic with the original
speed processor. It submits candidates when a sustained exceedance reaches the
existing duration and consecutive-frame gates. The original speed sample check
controls confirmation. The red-light adapter submits a directed stop-line
crossing for the original continuity and signal-history classification. The
lane-change adapter submits stable transitions and incomplete transition
diagnostics for the original restricted-boundary and reliability classification.
Legal, permitted, and uncertain outcomes are available in the candidate export.

Speed-only runs default to the original speed detector and tracker profile.
Runs with either geometry engine use the existing shared RLVD/LCVD profile
(detector confidence .15, ByteTrack high .35, low .14, new-track .50). The
session records which settings were used. Engine confirmation thresholds are
independent of this choice.

A candidate carries a unique event ID, source and session IDs, track and crossing
details, recent observations, applicable rule inputs, and evidence frame
references. The System A verifier delegates to the original rule. The engine
commits its original episode state independently of the verifier's disposition.
Confirmed events use the candidate's event ID and add plate results and the
session video reference.

All three violation types use one session-local plate scanner. It samples
native-resolution crops using the shared ByteTrack IDs and runs OCR on ranked
plate crops when a violation is confirmed. The Roboflow checksum restores the
legacy 960-pixel plate detector profile and PP-OCRv4/EasyOCR comparison. Other
videos default to EasyOCR. Results preserve absolute plate coordinates and
their source frame, detection confidence, OCR confidence, registration status,
and failure details. The existing registration validator and agreement or
high-confidence rules determine when a number can be shown as recognized.
Other OCR candidates remain unverified. OCR failures do not discard detected
plates or confirmed violations.

The annotated video shows red vehicle boxes for confirmed violations for
`anpr.violation_display_seconds` seconds (default 3). Yellow plate boxes
are drawn only on frames with a detector result for the same track. This
avoids displaying a stale box on a moving vehicle. The plate sampling interval
is controlled by the applicable ANPR profile. Registration text is shown when
recognized. The dedicated legacy all-track scan and OCR exports remain
available from the individual workflows.

SQLite stores sessions, candidates and confirmed events in
`outputs/unified/violations.sqlite3`. Each session keeps the annotated H.264
video, track CSV and evidence images. The unified results show the annotated
video and, for each confirmed event, the original and annotated violation
frame, vehicle crop, and raw and enhanced plate crops when a plate is
available. The plate source frame is shown when available. Earlier CSVs and evidence stay in their
existing directories and are accessible in the original screens or legacy
report categories.

### Test each violation type

| Selection | Scene setup | What to inspect |
| --- | --- | --- |
| Overspeed | Upload `dataset/roboflow/vehicles.mp4` and select **Matching video profile**, or set four ordered road corners and known width and length for another video. Set the speed limit. | Calibration label, sustained exceedance and the evidence frame. Roboflow and custom road dimensions produce estimated speeds. Reference profiles with a road polygon retain the original ROI gate. |
| Red light | Upload an intersection clip showing both the signal transition and vehicle approach. Draw the lamp ROI and stop line, select the side vehicles approach from, and optionally draw an approach polygon. | Signal history, crossing direction, confirmed red onset and crossing evidence. |
| Lane change | Upload a constant-frame-rate roadway clip. Draw at least three boundaries, including road edges, set the road orientation, lane numbering and monitoring direction, and mark interior dividers restricted or permitted. | Source and target lanes, the boundary classification and before/crossing evidence. |
| Multiple types | Select speed and lane change on a compatible road clip, or select other types where the scene provides their geometry. | Inspect track IDs, event types and evidence in a single saved session. |

Run the project regression suite with `python -m pytest tests -q`. The unified
integration tests use real video decoding and ByteTrack, controlled vehicle
detections, and the real violation rules. They exercise each engine alone,
all three together, rule dispositions, invalid geometry, OCR failure, storage,
and report exports. The Streamlit test checks the default workspace, conditional
controls, and report navigation.

### Verification results

The complete project suite passed **150 tests** with one dependency deprecation
warning. The real-footage runs below used the local YOLOv11 weights and
ByteTrack on CPU. The mixed run used a manually marked left carriageway and
treated the dashed divider as permitted.

| Video and selection | Frames | Detections | Tracks | Observed outcomes | Processing FPS | Mean frame latency |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Roboflow vehicles, speed only | 538 | 1,375 | 15 | 9 overspeed alerts from the reference calibration | 5.82 | 170.8 ms |
| Roboflow vehicles, speed and lane change | 538 | 1,833 | 11 | 10 estimated-speed alerts and 1 permitted transition candidate | 5.13 | 194.2 ms |
| ForeSpeed Kasa T1P1, speed only | 900 | 923 | 3 | Tracks available. Speed evaluation unavailable for this camera. | 13.92 | 71.6 ms |

The referenced evidence files were verified and the three annotated outputs
were decoded. The 19 Roboflow alert records returned unreadable plate text.
Red-light confirmation was exercised on a controlled clip with a visible
green-to-red change. The matching original intersection recording for the
saved configuration was unavailable in this workspace. These runs provide
processing measurements. Accuracy metrics require labelled ground truth.

### Implementation files

- Added `modules/unified/{__init__,pipeline,engines,schema,storage,dashboard}.py`,
  `modules/speed_detection/speed_session.py`, and `tests/test_unified.py`.
- Updated `app.py`, `modules/dashboard/{components,pages}.py`,
  `modules/speed_detection/{video_processor,violation_detector}.py`,
  `modules/rlvd/violation_engine.py`, `modules/lcvd/evidence_generator.py`,
  and this README.

### Cleanup and recovery

The full suite passed 150 tests before and after cleanup. Root-level backup
ZIPs are optional for running the app. `cleanup-recovery.zip` contains the
pre-cleanup versions of edited files and the removal manifest. Saved sessions
and evidence remain under `outputs/`.

### Experimental comparison

The experiment export records video identity, geometry and configuration, candidate
and confirmed counts, processing FPS, and mean frame latency in milliseconds.
FPS uses elapsed processing time through annotated-video encoding after model
initialization. Mean latency covers the frame loop, including per-frame I/O and
event work. Label-dependent metrics are null until there are annotated outcomes
and an agreed matching protocol. Define the unit of a negative example before
computing the false-positive rate.

A future System B can choose a verifier through an experiment configuration
and inject it at `process_video(..., verifier=...)`. It can inspect the candidate
and evidence, then delegate to the rule or reject the candidate. The built-in
rule still determines whether an event satisfies the original checks. Keep the
video, model weights, scene geometry, tracker profile and candidate rules
fixed. Match candidate exports across runs using video ID, rule type, track ID,
and frame number. Use the session and event IDs to locate individual artifacts.

Lane-change processing retains the existing constant-frame-rate requirement. The optional monitoring direction uses movement along the road at the observed boundary crossing.
The annotated output uses nominal playback FPS, while track CSVs retain valid
source timestamps. The supplied saved signal and lane geometries require their
matching source clips. Evaluation of speed or OCR accuracy and scene-level
precision/recall requires labelled data. Unified setup reuses saved geometries
when the source matches and stores applied geometry with each run.


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

