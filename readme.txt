What's required:
 - Python
 - numpy, scipy, matplotlib

My directory structure:
LWD/  --- local working directory
	 ohman/  --- sub-directory with git clone
	 phases_0.1_10_25/  --- example data set directory
	 ...  --- more of them

Example --- scripts are run from LWD:

$ ohman/gen_data.py gen_phases 0.1 10 30
# phases_0.1_10_30 subdirectory with two files (data and truth) is created

$ ohman/adj_graph.py phases_0.1_10_30 --K 4 --t 2.0
# directory phases_0.1_10_30/4_2.0 is created with more files:
# D  L  W  efuns  evals

$ ohman/show_efuns.py --truth phases_0.1_10_30/4_2.0 0
# this will show graph visualization (gray_r coded edge weights)
# along with eigenfunction 0 (first one, seismic encoded in vertex color)

# Please note that show_graph.py is not yet aligned with new data structure.
# Not so interestig... Anyway easy to upgrade.

