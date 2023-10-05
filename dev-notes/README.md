# Git Tree structure

The general git tree should be as follows:
```
main
└───dev
│   └───feature_branch_1
│   └───feature_branch_2
│   └───feature_branch_3
│   ...
│   └───feature_branch_n
```

`main` – only the production code, namely, bug-free code with all requirements specified in requirements.txt (remote branch)

`dev` – testing environment (remote branch):
- code needs some tests
- requirements are not complete
- etc.

`feature_branch_*` – development environment (local branches only)

# Process Flow

Create a branch for your feature:

`git pull origin main` – pull the latest code from the repo

`git checkout -b feature_branch_name` – create your feature branch

Feature branch naming should follow exact feature you are working on (feature implies any task). If you are working on EDA than the git command will look something like:

`git checkout -b exploratory_data_analysis`
