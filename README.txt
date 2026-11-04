1. Copy your "assets" folder (all the .obj files) into the "ur3e" folder, next to ur3e.xml.
2. Open a terminal in this folder and run:
     pip install mujoco opencv-python numpy
     python main.py --headless --trials 100      (or: python main.py   to open the viewer)
Layout should be:
  main.py, config.py, ... (the .py files)
  ur3e/ur3e.xml
  ur3e/assets/*.obj
