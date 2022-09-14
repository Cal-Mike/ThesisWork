Multi-Domain Profiling of Cyber Threats in Large-Scale Networks
Michael Calnan

*****STRUCTURE*****
-classifier
    |
    -datasets (contains all datasets created from dataExploration.py)
        |
        -training (July 14th data used to train models)
        -validation (July 15-20th data used to validate models)
    -encoders (ColumnTransformer object pickle)
    -models (contains saved models for later use)
    -results (text and picture files for each test)

-cluster
    |
    -dataframes (datasets used for clustering from July 14th)
    -encoders (ColumnTransformer object pickle)
    -models (saved mini-batch K-means object pickle)
    -results (text and .csv files from tests)

*****RUNNING INSTRUCTIONS*****
"./submit_gpu.sh" or "./submit.sh" on Hamming HPC. If using "gpu" options, must have access to the partitions described in the SBATCH options header fields.

*****HOW TO GENERATE DATASETS*****
Classifer:
-Ensure dataExploration.py paths point to pandas dataframes. 
-Change read_file() internal filename variable to the specified time and date.
-Run the dataExploration.py file using the BASH files described in RUNNING INSTRUCTIONS.

Cluster:
-Ensure dataExploration.py paths point to numpy array files.
-Ensure pyarrow or fastparquet libraries are installed via pip.
-Change filename variable in main method.
-Run file via RUNNING INSTRUCTIONS.
