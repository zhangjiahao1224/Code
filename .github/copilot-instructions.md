---
name: copilot-instructions
description: |
  Workspace-wide Copilot instructions for the d:\\Code robotics/machine-learning repository.
  Includes structure, standard tasks, conventions, and prompt samples.
  Use this file as the root guidance for agent behavior and code quality checks.
applyTo: "**/*"
---

## 1. Project Overview

- Root folder `d:\Code` contains multiple language areas:
  - `C++/` (ROS, Algorithm, Libs, Projects, Learning path)
  - `Py/` (DataAnalysis, Scripts, Projects, Learning, LeetCode, LeRobot)
  - `Common/` (Configs, Datasets, Docs)
  - `Matlab/` (Robotics, DataAnalysis)
  - `Archive/` for older experiments and external repos.
- Primary focus for current work is in `Py/Projects` and `C++/Projects`.
- Code uses mixed Chinese/English comments and docs; maintain readability while adopting English in new shared APIs.

## 2. Build & Run

### C++
- `g++` compile quick file:
  - `g++ ${file} -o ${fileDirname}/${fileBasenameNoExtension}.exe -static -Wall -g`
- Use `C++/ROS` for ROS topics and launch files.
- Prefer `CMakeLists.txt` in complete projects.

### Python
- Conda environment: `conda activate pytorch` (existing terminal context).
- Python version: `>=3.8`.
- Test script path: `Py/Projects/pytorch/lesson3.py`.

### Matlab
- Use `R2020b+`.

## 3. Productivity conventions

- Keep changes small and focused by directory.
- Add TODO comments for unfinished work.
- Include a short Chinese/English summary in PR descriptions when helpful.
- For bugfixes, include repro steps and expected behavior.

## 4. Code recommendations for Copilot

- Use existing utilities in `utils.py` and `Py/Utils` where applicable.
- Reuse patterns in `Py/Learning/0X-*` when adding educational examples.
- For deep learning code, follow patterns in `Py/Projects/pytorch` (data loader, model, train loop, eval). 

## 5. Suggested prompt templates

- "Search for TODO and missing test coverage in `Py/Projects/pytorch`."
- "Refactor `C++/Algorithm` function `xxx` to avoid raw pointers and use smart pointers."
- "Add docstring and type hints for Python function in `Py/LeetCode/L1.py`."

## 6. On extending this file

If there are workspace-specific engine/toolchain steps (Docker, GPU path, ROS distro), add them below with explicit paths and terminal commands.

---

> Note: This is intentionally minimal so it does not over-claim behavior, and it is easy to update as repository conventions stabilize.
