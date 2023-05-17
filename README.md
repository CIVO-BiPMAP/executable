# CIVO BiPMAP - executable version
## For users
### Prerequisites 
General:
1. Windows OS (x64), 16gb RAM (more is recommended)
3. Install Python3 (tested on version 3.10) and add it to PATH.

For GPU Acceleration:
1. GPU must be CUDA supported with over 8gb memory (tested with 48gb).
2. CUDA must be installed (tested on version 11.7).
### Installation
1. Download the [latest release](https://github.com/CIVO-BiPMAP/executable/releases) of BiPMAP and unzip to your Documents.
4. Inside the folder, launch BiPMAP.exe (recommended to pin to taskbar).

**During the first launch, BiPMAP automatically installs libraries which takes a few minutes.**

### Troubleshooting
Let us know if you experience [issues](https://github.com/CIVO-BiPMAP/executable/issues).

Installation logs are available when running BiPMAP.exe from the command line.

Note that CUDA out of memory issues are hard to diagnose and may present themselves under different names. When in doubt, reducing the "Recording Length" parameter will reduce the memory footprint.
