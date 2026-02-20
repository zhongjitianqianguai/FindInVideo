# AGENTS.md - Agentic Coding Guidelines for FindInVideo

## 语言要求
- **必须使用中文回答所有问题**
- 代码注释和 print/log 消息使用中文
- Docstring 内容使用中文（如 `"""检查文件是否为视频文件"""`）
- AGENTS.md 本身使用中文

## Project Overview
FindInVideo is a Python tool that uses YOLO (v11 and v5) to detect objects in videos and record their time positions. It processes directories of video files, runs object detection frame-by-frame, and writes timestamp artifacts alongside each video. The codebase runs on Windows with some WSL/UNC path compatibility.

## Build & Run Commands

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r yolov5/requirements.txt
pip install ultralytics         # For main.py (YOLOv11)
```

### Running
```bash
python main.py              # YOLOv11 entry point (recommended)
python main_yolov5.py       # YOLOv5 entry point
python estimate_remaining_time.py --root D:\z   # ETA utility
```

### Testing
There are currently no tests. If tests are added, use pytest:
```bash
pytest                                      # Run all tests
pytest test_file.py                         # Single test file
pytest test_file.py::test_function_name     # Single test function
pytest --cov=. --cov-report=term-missing    # With coverage
```

### Linting (not currently configured, but recommended)
```bash
pip install ruff black
black .
ruff check .
ruff check . --fix
```

## Project Structure
```
FindInVideo/
├── main.py                     # Main entry, YOLOv11 (ultralytics), ~1556 lines
├── main_yolov5.py              # YOLOv5 entry point, ~1493 lines
├── main_restored.py            # Non-functional decompilation artifact (ignore)
├── estimate_remaining_time.py  # CLI tool for processing time estimation
├── yolov5/                     # YOLOv5 git submodule (external code)
│   ├── detect.py, train.py, ...
│   ├── requirements.txt
│   └── utils/, models/, data/
├── models/                     # YOLO model weights (.pt files)
├── md5_list/                   # Processed-files MD5 tracking
├── logs/                       # Runtime logs (crash.log, run.log)
├── training_data/              # Training data output
└── run_main.bat, run_main_yolov5.bat  # Windows launchers
```

Note: `main.py` and `main_yolov5.py` contain significant duplicated code (path utils, MD5, checkpoint system, artifact naming, directory traversal).

## Code Style Guidelines

### Imports
- Standard library first, third-party second, local last
- Blank lines between groups are not strictly enforced but preferred
```python
import os
import sys
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

from yolov5.utils.general import non_max_suppression
```

### Formatting
- 4 spaces indentation
- No enforced line length limit (lines often exceed 120 chars)
- **Single quotes** for strings in practice (the codebase predominantly uses `'string'`)
- Double quotes appear in f-strings and some literals; either is acceptable

### Naming Conventions
- Functions/variables: `snake_case` (e.g., `detect_objects_in_video`)
- Classes: `PascalCase` (e.g., `DirectoryIndex`, `PauseRequested`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `VIDEO_EXTENSIONS`, `ARTIFACT_SUFFIXES`)
- Private: `_leading_underscore` (e.g., `_truthy_env`, `_read_frame_with_timeout`)

### Type Hints
- Not widely used in existing code. Adding them to new or modified functions is encouraged but not required to match existing style.

### Comments and Strings
- **Chinese** is the primary language for inline comments and print/log messages
- English is used for AGENTS.md and README
- Docstrings use triple double quotes, content often in Chinese (e.g., `"""检查文件是否为视频文件"""`)
- Keep comments current with the code

### Error Handling
- Use specific exceptions where possible; `except Exception:` with `pass` is common for best-effort cleanup
- Custom exception: `PauseRequested` for graceful checkpoint-and-stop
- Always release resources (VideoCapture, DB connections) in `finally` blocks
```python
try:
    cap = cv2.VideoCapture(path)
    # ... process frames ...
except Exception as e:
    print(f"处理出错: {e}")
finally:
    if cap:
        cap.release()
```

### Memory Management
- Call `gc.collect()` periodically during large video processing
- Explicitly `del` large arrays/tensors when done
- Release `cv2.VideoCapture` and `cv2.VideoWriter` in `finally` blocks

## Key Constants
- `VIDEO_EXTENSIONS`: Set of 21 recognized video file extensions
- `ARTIFACT_SUFFIXES`: `['_frames.mp4', '_objects.mp4', '_detections.mp4', '_mosaic.jpg', '.txt', '.done']`
- `CHECKPOINT_SUFFIX`: `'.checkpoint.json'`
- `DONE_SUFFIX`: `'.done'`

## Environment Variables (prefix: `FINDINVIDEO_`)
| Variable | Description | Default |
|---|---|---|
| `FINDINVIDEO_SHARED_STATE_DIR` | Shared state dir for multi-instance coordination | (none) |
| `FINDINVIDEO_PAUSE_FILE` | Path to pause flag file | (auto) |
| `FINDINVIDEO_RESUME` | Enable checkpoint resume | `true` |
| `FINDINVIDEO_IMGSZ` | Model input image size (main.py only) | `1920` |
| `FINDINVIDEO_YOLOED_PATH` | Path to processed-files MD5 list (main.py) | `md5_list/yoloed.txt` |
| `FINDINVIDEO_YOLOED_PATH_YOLOV5` | Same for main_yolov5.py | (auto) |
| `FINDINVIDEO_WIN_TEMP` | Windows temp dir for MD5 workarounds | (none) |
| `FINDINVIDEO_CLAIM_TTL_SECONDS` | TTL for video claim locks | `86400` |

## Database (SQLite)
- File: `directory_index.db` in script directory, managed by `DirectoryIndex` class
- WAL mode enabled, foreign keys on
- Tables: `directories` (path, parent_path, mtime, scan info, flags) and `videos` (dir_path, file_name, mtime, size, is_video)
- Global singleton `DIRECTORY_INDEX` cleaned up via `atexit.register`
- Always use parameterized queries

## Cross-Platform
- Uses `os.path` for path operations; avoid hardcoded separators
- Path conversion utilities exist for Windows drive paths, WSL `/mnt/` paths, and UNC `\\wsl.localhost\` paths
- Windows long-path issues are handled in several places

## Key Dependencies
- `ultralytics` (YOLOv11 in main.py)
- `opencv-python` (video I/O)
- `numpy`, `torch` (inference)
- `tqdm` (progress bars)
- YOLOv5 dependencies via `yolov5/requirements.txt`

## Git
- Branch: `master`
- Commit messages: often in Chinese, sometimes prefixed with `feat:`
- `.gitignore`: `.venv/*`, `__pycache__/*`, `logs/*`
