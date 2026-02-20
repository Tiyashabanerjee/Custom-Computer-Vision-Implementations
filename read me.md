READ ME FILE - 

You will find implementations for:

Straight line detection in binary images (q1).

Corner detection using Harris response (q2).

Line and circle detection using custom Hough Transform (q3).

Note: Built-in Hough functions (e.g., hough() in MATLAB) are not used. Almost all the implementations are written from scratch.

Directory Structure
ProjectComputer Vision_assignment 2/
└─ Data/
   ├─ Question1.png
   ├─ Question2-1.jpg
   └─ Question3/
       ├─ 3.png
       └─ train.png
│
├─ q1/
│  ├─ main.py
│  └─ outputs/
│  └─_init_.py

├─ q2/
│  ├─ main.py
│  └─ outputs/
│  └─_init_.py
├─ q3/
│  ├─ main.py
│  └─ outputs/
│  └─_init_.py
├─ utils.py
├─ run_all.py
├─ raad me.md


DEPENDENCIES

1. Install the following Python packages before running the project:
2. pip install numpy scipy matplotlib scikit-image

How to Run

1. To generate all outputs for q1, q2, and q3, just run:
python run_all.py

This will execute all three components sequentially.

NOTE: Output images will be saved in their respective outputs/ folders:

q1/outputs/

q2/outputs/

q3/outputs/

Check these folders for gradient maps, edge maps, detected lines, corners, and circles.