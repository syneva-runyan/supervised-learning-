# README.txt

The code for this assignment can be found in the [public repository here](https://github.com/syneva-runyan/supervised-learning-)

The `algorithms.py` file can be ran to execute the assignments algorithms on both data sets.

In order to run the algorithms, you must first install each of the modules imported at the top of the file using pip

```pip install sklearn
pip install pandas
pip install matplotlib
pip install time
pip install numpy```

From there, you can run the algorithms with:

`python algorithms.py`

Each algorithms will then run against the World Cup prediction data set first,
and then each algorithm will run agains the Heart Failure Prediction data set.

In alot of implementations, scikit code was referenced. 
This was done especially so for the [decide_ccp_alpha example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py)