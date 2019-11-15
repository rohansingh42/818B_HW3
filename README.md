# HW3 : Q LEARNING

## Dependencies

Written on Python 3.5
- Tensorflow 1.14 (CPU only)
- OpenAI Gym
- numpy
- matplotlib

## Execution

Open Terminal from base directory (818B_HW3_TSP)

For training,

```
python3 problem1_sol.py --train=True
```

For testing,
```
python3 problem1_sol.py
```
## Solution Description

- The tensorboard graphs are generated in the logs folder.
- The models are stored in the checkpoints folder. There are 2 trained models with slightly varying performance because of random initializations (and randomly finding successful episode early on). To test either, change the path in the python sript.