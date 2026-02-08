# Works for application
## Topic: <br>&ensp;*Rl for locomotion sim2realgap energy-efficient locomotion,hierarchical systems,and physically-consistent pipelines*
## My main references (reviews):<br>
**[1]**.W. Zhu, X. Guo, D. Owaki, K. Kutsuzawa and M. Hayashibe, <<A Survey of Sim-to-Real Transfer Techniques Applied to Reinforcement Learning for Bioinspired Robots,>> in IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 7, pp. 3444-3459, July 2023, doi: 10.1109/TNNLS.2021.3112718.
https://ieeexplore.ieee.org/document/9552429

**My conclusions and understandings about it:**

**Solving the simulation-to-real problem needs a combination of methods.**

i.Better simulators.Build a suitable training environment for RL

&ensp;accurate robot parameters.
 
&ensp;trains use Domain Randomization for robustness

ii.Better Models 


&ensp;Models of Motor Primitives

&enspModels of State Transition

iii.Hierarchical Controllers and Distributed Controllers

iV.examples

**V.Conclusions, Outlook, and Limitations**

&ensp;Deploying Relay (RL) on complex robots is challenging; 
&ensp;the stability of RL needs improvement
&ensp;Better simulators are required to adapt RL for deployment on real robots.

**Following this paper,i found some related papers,However, the problem is that many related papers solving problems in this field do not contain any open-source code.**

**such as**

Pan, Jia-Hao, et al. "Deep reinforcement learning based optimization of microwave microfluidic sensor." IEEE Microwave and Wireless Technology Letters (2024).

Cao, Pan, et al. "Computational intelligence algorithms for UAV swarm networking and collaboration: A comprehensive survey and future directions." IEEE Communications Surveys & Tutorials 26.4 (2024): 2684-2728.

Liu, Lixing, et al. "A reinforcement learning path following strategy for snake robots based on transferable constrained-residual gait generator." IEEE Transactions on Industrial Electronics 71.12 (2024): 16013-16025.

Ye, Neng, et al. "Artificial intelligence for wireless physical-layer technologies (AI4PHY): A comprehensive survey." IEEE Transactions on Cognitive Communications and Networking 10.3 (2024): 729-755.

Qi, Ji, et al. "Reinforcement learning and sim-to-real transfer of reorientation and landing control for quadruped robots on asteroids." IEEE Transactions on Industrial Electronics 71.11 (2024): 14392-14400.

Zhang, Tianhao, et al. "Leveraging imitation learning on pose regulation problem of a robotic fish." IEEE Transactions on Neural Networks and Learning Systems 35.3 (2022): 4232-4245

Zhu, Wei, and Mitsuhiro Hayashibe. "Autonomous navigation system in pedestrian scenarios using a dreamer-based motion planner." IEEE Robotics and Automation Letters 8.6 (2023): 3836-3843.

Li, Zhiming, et al. "Learning-based data gathering for information freshness in UAV-assisted IoT networks." IEEE Internet of Things Journal 10.3 (2022): 2557-2573.

Zhu, Wei, and Mitsuhiro Hayashibe. "A hierarchical deep reinforcement learning framework with high efficiency and generalization for fast and safe navigation." IEEE Transactions on industrial Electronics 70.5 (2022): 4962-4971.

Dowdy, Jordan, and Jean Chagas Vaz. "Isaac Sim-to-Real: Reinforcement Learning based Locomotion for Quadrupeds." 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE). IEEE, 2025.

Chaoxia, Chenyu, et al. "Sim-to-real transfer of automatic extinguishing strategy for firefighting robots." IEEE Robotics and Automation Letters (2024).

After a brief reading, I found that ***none of them had open-source code***.

Then I then found a website that allows me to better view the relevance of related papers, using graphs for easier reference: Connected Papers https://www.connectedpapers.com/

![image](connectedpapers.png)

Then I found 3 papers，they have codes that looks like I can reproduce:

**[2]**.Cheng X, Shi K, Agarwal A, et al. Extreme parkour with legged robots[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024: 11443-11450.

**[3]**.Zhuang Z, Fu Z, Wang J, et al. Robot parkour learning[J]. arXiv preprint arXiv:2309.05665, 2023.

**[4]**.Smith L, Kew J C, Li T, et al. Learning and adapting agile locomotion skills by transferring experience[J]. arXiv preprint arXiv:2304.09834, 2023.

To prevent the code from becoming outdated or incorrect, I selected 3, 2 of which can be used as an alternative.

firstly i choose Zhuang Z, Fu Z, Wang J, et al. Robot parkour learning[J]. arXiv preprint arXiv:2309.05665, 2023.

Its open-sourced code in https://github.com/ZiwenZhuang/parkour/tree/main

#### System and environment configuration before the experiment：

I installed the Ubuntu Linux system on my computer using a USB drive.

Install the latest version of Python.

```
sudo apt update && sudo apt upgrade -y
sudo apt install python3
```

Install isaacgym:

Unzip the compressed file isaacgym

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
restart shell

```
~/miniconda3/bin/conda init bash
```

restart shell

```
~/miniconda3/bin/conda init zsh
```

```
cd /<my document...>/isaacgym/python/
bash ../create_conda_env_rlgpu.sh
```
run virtual environment

```
conda activate rlgpu
```

```
pip install -e .
```

Running Example

```
cd examples 
python joint_monkey.py
```

![image](isaacgym.png)
