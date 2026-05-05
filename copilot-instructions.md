\# IMPORTANT: GVSU CLIPPER HPC RULES FOR THIS PROJECT



\## MANDATORY PRE-FLIGHT CHECK — Every Single Time



Before executing ANY command (other than the allowed list below), you MUST run:



```bash

hostname



Verify the output is NOT port and NOT port.clipper.gvsu.edu.



If output contains port:



You are on the login node.



STOP immediately.



Do NOT run compute-heavy commands.



Do NOT run preprocessing over the dataset.



Do NOT run model training.



Do NOT run evaluation loops.



Obtain a compute node first or submit a batch job.



If output is like b004, c001, g001:



You are on a compute node.



Proceed.



This check is NOT optional. Run it every time, even if you think you are already on a compute node.



ABSOLUTE RULES



NEVER do compute-heavy work on the Port/login node.



ALWAYS use Slurm batch submission for experiments.



ALWAYS prefer sbatch for reproducible runs.



The login node is ONLY for:



editing files



checking paths



creating virtual environments



installing packages



submitting jobs



inspecting logs/results



DO NOT run training on the login node.



DO NOT run full-dataset preprocessing on the login node.



DO NOT run LOSO evaluation on the login node.



DO NOT run MiniROCKET fitting on the login node.



DO NOT interrupt jobs unless necessary.



ALLOWED SAFE COMMANDS ON LOGIN NODE



hostname



pwd



ls



cd



mkdir



cp



mv



rm (carefully)



cat



less



head



tail



nano



vim



python --version



pip install ...



python -m venv ...



source .venv/bin/activate



sbatch ...



squeue



sinfo



If a task requires real computation, write or update an sbatch script and submit it.



REQUIRED WORKFLOW



Run hostname



Confirm you are not on port



Check file paths and outputs



Prepare or update sbatch script



Submit with sbatch



Monitor with squeue



Inspect logs with tail -f or less



Save outputs to structured directories



Summarize results in markdown



PROJECT GOALS



This project is a reproducible NinaPro DB3 EMG gesture-classification benchmark with strict LOSO evaluation.



Primary goals:



run MiniROCKET LOSO correctly on HPC



compare against existing baselines



run at least one simple deep neural network baseline



measure cross-subject generalization



report balanced accuracy as the primary metric



MODELING REQUIREMENTS



Final layer of any deep neural network must be a SoftMax classifier for multiclass classification.



In PyTorch, models may output logits and use CrossEntropyLoss, which is equivalent for training.



Use the same preprocessing and LOSO harness for all methods.



REQUIRED OUTPUTS FOR EACH METHOD



For every method, produce:



per-subject outputs



merged CSV summary



balanced accuracy and accuracy



plots



markdown notes containing:



concept



math



algorithm



success metrics



results



discussion



conclusion



references

