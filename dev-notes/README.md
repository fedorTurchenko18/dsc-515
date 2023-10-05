# Git Tree structure

The general git tree should be as follows:
```
main
└───dev
    └───feature_branch_1
    └───feature_branch_2
    └───feature_branch_3
    ...
    └───feature_branch_n
```

`main` – only the production code, namely, bug-free code with all requirements specified in requirements.txt (remote branch)

`dev` – testing environment (remote branch):
- code needs some tests
- requirements are not complete
- etc.

`feature_branch_*` – development environment (local branches only)

# Process Flow

Create a branch for your feature:

## Step 1

`git pull origin main` – pull the latest `main` code from the repo

## Step 2

Pull the latest `dev` code from the repo:

```
git checkout dev
git pull origin dev
```

## Step 3

`git checkout -b feature_branch_name` – create your feature branch

Feature branch naming should follow exact feature you are working on (feature implies any task). If you are working on EDA than the git command will look something like:

`git checkout -b exploratory_data_analysis`

## Step 4

Once you have finished working on your feature branch (i.e. commited all the changes), merge it with the `dev` one and push results to the remote:

```
git checkout dev
git merge feature_branch_name
git push origin dev
```
