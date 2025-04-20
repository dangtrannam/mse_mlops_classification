# Active Context

## Current Work Focus
- Initial project setup: directory structure, requirements, and memory bank files.

## Recent Changes
- Created all main project directories.
- Added requirements.txt with all dependencies.
- Created memory bank files: projectbrief.md, productContext.md, techContext.md, systemPatterns.md.

## Next Steps
- Create progress.md to track ongoing status.
- Begin implementation of data generation module (src/data_utils.py).

## Active Decisions
- Using local MLflow tracking (`./mlruns`).
- PyTorch MLP with 1-2 hidden layers as the model.
- Flask for deployment with basic HTML UI.
- Register best model to "Staging" in MLflow Model Registry. 