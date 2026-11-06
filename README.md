This project implements Track B (Manipulator: Vision-Guided Pick) using a UR3e robotic arm in MuJoCo with a wrist-mounted RGB-D camera.

The system detects a red cube on a table, estimates its 3D position using image and depth information, and picks it up using Position-Based Visual Servoing (PBVS).

The cube position is estimated once from the camera image and depth data, transformed into the world coordinate frame, and converted into joint targets using damped least-squares inverse kinematics (DLS IK).

To evaluate robustness, the cube position is randomised on every trial, the cube slides across the table before coming to rest, and sensor measurements contain noise. The system is designed to localise and pick up the cube despite these variations.

Contribution
Part A: Dija
Part B:Alzaid Muhammed 
Part C:Do Nguyen Khoi Nguyen 
Part D: Alzaid Muhammed, Leon Markovic 
Part E: All members
 
