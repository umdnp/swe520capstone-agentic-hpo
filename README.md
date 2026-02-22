# SWE 520 Capstone Project - Agent-based HPO for Federated Learning

This capstone project builds on prior work comparing centralized and federated learning, shifting the focus
to a more realistic federated setting where data cannot be centrally inspected or interactively explored.
In healthcare, data are often distributed across institutions and cannot be pooled for direct analysis.
Under these constraints, hyperparameter optimization (HPO) in federated learning cannot rely on interactive
inspection or immediate feedback. Instead, tuning decisions must be made using delayed, aggregated metrics,
raising a central question: is federated HPO better understood as a fixed configuration choice made before
training, or as a sequential decision process that unfolds over time?

This project evaluates whether agent-based control is well-suited to this type of optimization. Rather than
assuming agents improve performance outright, the goal is to examine whether they can adapt hyperparameters
during training using limited global signals and make more consistent decisions than manual or static approaches.
Experiments use the eICU Collaborative Research Database (eICU-CRD), a multi-center critical care dataset
representative of real-world data-driven healthcare settings where federated learning may be necessary. 
The emphasis is not on maximizing benchmark accuracy, but on understanding whether agent-driven HPO provides
a practical and structurally sound approach under real federated constraints.

**Author:**
- Jim Prantzalos
