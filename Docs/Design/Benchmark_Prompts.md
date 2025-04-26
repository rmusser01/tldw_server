# Prompts to Benchmark Models in random stuff




Tree burning:
https://www.reddit.com/r/LocalLLaMA/comments/1k1nle9/inspired_by_the_spinning_heptagon_test_i_created/
```
Write a Python program that simulates the spreading of a forest fire:
- use complex shapes for the trees so that they are as realistic as possible. Color the trunk and branches brown and the leaves green.
- use a side-view 2D approximation
- use at least 10 trees to simulate the forest
- color the fire as a dark orange
- you must visibly show the flames as they propagate as well as any embers that move and light new trees on fire
- the fire should start near the center of the trees and spread according to the wind
- simulate a 20mph wind going from the left-to-right
- make the initial conditions appropriate such that the fire propagates and burns at least 40% of the trees
- Do not use the pygame library; implement all needed physics by yourself. The following Python libraries are allowed: tkinter, math, numpy, dataclasses, typing, sys, opencv-python
- you MUST output an .mp4 video file with a resolution of 1280x720 of the simulation after it completes and adjust the speed of the video so that the video duration is exactly 30 seconds. Name the .mp4 file as *your_AI_name*.mp4
- you will be judged on how accurate and visually appealing the simulation is so make it look amazing
- all code should be put in a single Python file.
```