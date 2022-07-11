README for deCODE

1. prerequisites
deCODE is writen and tested with Python 2.7.10 on RHEL6. It should run on any
Linux system that supports Python and its dependencies. It requires the 
following python packages:

h5py	2.5.0
keras	1.1.1
matplotlib	1.5.3
networkx	1.10
numpy	1.11.2
pandas	0.19.1
scipy	0.18.1
sklearn	0.17
theano	0.9

These packages can be installed using pip.
deCODE requires at least 4GB memory to run properly.

2. input file format

deCODE takes tab-delimited gene expressions for a group of samples as input, where 
the rows are genes and columns are samples. The gene expression should be 
normalized and log2 transformed. Please refer to the sample data as used in the 
deCODE paper for detailed format. The number of samples, as used in the deCODE paper, 
should be at least about 100. A portion of the sample data looks like the 
following:

gene_symbol	GSM615096	GSM615097
AKT3	0.0484362055555555	0.119655650740741
MED6	0.20134624	0.0812970794117647
NR2E3	-0.141484470625	-0.18230819625	-0.24029588875


3. workflow

To run deCODE, please call the main python script with a project name and a data file:
DECODE_STEP=0 python decode_main.py <project_name> <input_file_name>
e.g.
DECODE_STEP=0 python decode_main.py RD GSE25066_RAW_scan_symbol_samplename_RD.txt

The project name can only contains letters, numbers and underscore ("_"). The 
command can only be called in the same folder where data file resides.

4. output

config_decode_<project_name>.rawscore.ntop200.gml
  The network of top 200 gene-gene associations given in gml format. This can be
  visualized using network visualization software such as Cytoscape.

config_decode_<project_name>.rawYBscore.inputtomid.h5
  Weights from hidden layer nodes to every input genes. These can be used to run
  GSEA-preranked to explorer the function annotation of learned compact
  representation of the input data.


2022 Jul 10
