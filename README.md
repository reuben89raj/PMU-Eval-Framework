# PMU-Eval-Framework
Framework to evaluate resiliency of PMU networks

To execute a particular design provide the following command:

python3 <filename.py> <num_links> <search_alg> > /tmp/out

Examples:
1. python3 118bus-withRed.py 2links bfs > /tmp/out
2. python3 118bus-withoutRed.py 1link bfs > /tmp/out

Note:
1. 118bus files require corresponding topologies python file to load the design. Ensure topologies_1.py and topologies_2.py are placed in same folder as the design file
