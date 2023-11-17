### Snake - AI

Snake - AI is a project developed within a customized environment of the classic game Snake. Reinforcement Learning was implemented to teach the agent how to play.

The implementation of the agent was similar to other projects already presented in other repositories, following the official Gymnasium and Stable Baselines 3 documentation. However, in this project I practiced the elaboration of a customized environment adapted for a Snake game, developed in Python.

Also, in this project, I chose to experiment with a Python file, instead of using Jupyter, as usual.

The following links should redirect you to their respective documentation:

```Stable Baselines 3``` - https://stable-baselines3.readthedocs.io/en/master/

```Gymnasium``` - https://gymnasium.farama.org/index.html

##

If you want to run the Python files contained in this repository on a Windows operating system, or even customize the agent, I strongly recommend reading up on the peculiarities of compatibility with each operating system.

During development, ```Anaconda/Miniconda``` was used to manage the packages and work around incompatibilities.
- https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/windows.html

## 

To run this project, you need to execute the command ```python snake_ai.py``` inside the project's root folder. You also need to have installed the base packages from Gymnasium and Stable Baselines 3, described in the documentation provided above.

To check that the environment is compliant and functional, the file ```check_snake_custom_env.py``` can be found inside the ```snake_env``` folder. Running this file, in a similar way to the main file in the root folder, will check the environment using Gymnasium's ```Check``` tool. Any irregularities within the enviroment should be reported.

It is also possible to make changes to the environment, if desired. The environment file can be found in the same folder, named ```snake_custom_env.py```.

##

Monitoring and evaluation of the PPO agent was carried out using Tensorboard. You can load the agent's log files by following the documentation below:

```Tensorflow``` - https://www.tensorflow.org/tensorboard/get_started?hl=pt-br

The following graph represents the smoothed average of agent rewards over 3 Million Training Steps.

![image](https://github.com/Henrique-Bidarte/snake-ai/assets/134324510/14b01754-9fdd-471f-9112-e47e787e218a)

##

It is possible to control the project flow through the ```constants``` section in the ```snake_ai.py``` file. Not only it assigns values ​​to the different variables consumed, but it indicates whether the project should generate a new model for experimentation or train an existing one.

Logs will be saved within the ```logs``` folder so they can be analyzed accordingly.

Trained models are saved in the ```models``` folder alongside the number of steps executed in the training.

A ```best_model``` file can be found within the project root folder. It is an appropriate starting point and should be replaced periodically with better trained versions.

##

![Untitled2](https://github.com/Henrique-Bidarte/snake-ai/assets/134324510/d6065680-8ab3-484e-b299-8b6fa1540ada)
