## Setup Instructions
Conda is required to run the setup script included with this repository.
To avoid licensing issues with Anaconda, it is recommended you install conda on your machine via
[Miniconda](https://docs.anaconda.com/miniconda/) rather than Anaconda.

To create a fresh conda env with all the necessary dependencies, simply run
```
chmod +x setup.sh
bash setup.sh
```
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and install mujoco210.

You might need to install ffmpeg, with
```
sudo apt install ffmpeg
```

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run:
```
python test.py --all
```
Alternatively, you can run each test individually with the following commands:
```
python test.py --sim
python test.py --env
python test.py --algo
python test.py --nn
python test.py --render
python test.py --mirror
python test.py --timing
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.

## Evaluation Instructions
After training a policy (or you can test with the provided policies in `./pretrained_models`) you can evaluate with the `eval.py` script. For example, run
```
python eval.py interactive --path ./pretrained_models/CassieEnvClock/spring_3_new/07-12-14-27/
```

## Structure Overview

The repo is split into 6 main folders. Each contains it's own readme with further documentation.
- [`nn`](nn): Contains all of the neural network definitions used for both actors and critics. Implements things like FF networks, LSTM networks, etc.
- [`env`](env): Contains all of the environment definitions. Split into Cassie and Digit envs. Also contains all of the reward functions.
- [`sim`](sim): Contains all of the simulation classes that the environment use and interact with.
- [`testing`](testing): Contains all of the testing functions used for CI and debugging. Performance testing for policies will go here as well.
- [`util`](util): Contains repo wide utility functions. Only utilities that are used across multiple of the above folders, or in scripts at the top level should be here. Otherwise they should go into the corresponding folder's util folder.