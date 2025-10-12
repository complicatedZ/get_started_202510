FBAsolution
------------
Contains flux balance analysis (FBA) solutions under different experimental or simulated settings.
File names follow the format: M9_sufficient_M_N
- M indicates the number of provided nutrients.
- N indicates the number of different environments tested.

flow_graph_M
------------
Contains constructed mass-flow graphs derived from the corresponding FBA solutions.
M represents the number of nutrients provided in the medium.

flux_flow_graph_construct.ipynb
-------------------------------
An example notebook demonstrating how to construct a mass-flow graph from FBA solution files.

flux_flow_graph_union.ipynb
---------------------
Includes scripts and utilities for coarse-graining reaction networks
by integrating mass-flow graphs across multiple environmental conditions.
