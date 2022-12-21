Steps:
1. FrugalLight Training uses a traffic simulator called as CityFlow. So we first need to setup the CityFlow simulator.
2. ``git clone https://github.com/cityflow-project/CityFlow.git``
3. Go inside the City Flow directory, `` cd CityFlow ``
4. ``python3 setup.py build``  --> This will generate the .so file. Copy this .so file to the FrugalLight directory.
5. Or after building, run ``python3 setup.py install`` ---> In this case, city flow will be installed and you do not need to copy the .so file.
6. Then go to FrugalLight directory, run ``python3 code/runexp.py``

<b>NOTE:</b> You can go down to know more above the different files of FrugalLight code.

# FrugalLight

FrugalLight is a reinforcement learning agent for AI based separate and independent (without network communication) 
traffic signal control. It intends to match state-of-the-art-performance using low computational resources.

The reference code for this project is an open source project Presslight at github.

Start an experiment by:

``python3 code/runexp.py --transform=?``

where ? can be any prefix from Lane,Approach,Group,Relative.

## Datasets

  FrugalLight uses real data for the evaluations. Traffic file and road networks of New York City and New Delhi can be found in ``data``, it contains two networks of NYC at different scales (16 intersections and 48 intersections), and New Delhi data is for a 3-approach single intersection.

## Modules

* ``runexp.py``

  Run the pipeline under different traffic flows with various arguments/parameters. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``.

  For most cases, you might only modify traffic files and config parameters in ``runexp.py`` via commandline parameters.

* ``dqn.py``

  A DQN based agent build atop ``agent.py``

* ``config.py``

  The default configuration of this project. Note that most of the useful parameters can be updated in ``runexp.py`` via command line arguments.

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a simulator environment, run a simulation for certain time (one round), construct samples from raw log data, update the model etc.

* ``generator.py``

  A generator to load a model, start a simulator environment, conduct a simulation and log the results.

* ``anon_env.py``

  Define a simulator environment to interact with the simulator and obtain needed data.

* ``construct_sample.py``

  Construct training samples from data received from simulator. Select desired state features in the config and compute the corresponding reward.

* ``updater.py``

  Define a class of updater for model updating.
  
## Simulator

  The project uses an open source simulator CityFlow to get the impact of FrugalLight's actions in the environment. The default library is built for Python 3.6, and a Python 3.7 variant is also provided. If the given libraries don't work for you, please build the Cityflow simulator from source code.

## Disclaimer

The research paper on this project is under review, so keep this code confidential. If found public, please notify to github id sachin-iitd
