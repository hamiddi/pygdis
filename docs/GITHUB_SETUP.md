# Creating the GitHub repository

## Using the GitHub website

1. Sign in to GitHub.
2. Select **New repository**.
3. Name the repository `pygdis`.
4. Choose **Public** or **Private**.
5. Do not initialize it with a README, license, or `.gitignore`, because these files are already included.
6. Create the repository.
7. In this local project directory, run:

```bash
git init
git add .
git commit -m "Initial pyGDIS release"
git branch -M main
git remote add origin https://github.com/hamiddi/pygdis.git
git push -u origin main
```

## Using GitHub CLI

```bash
gh auth login
gh repo create pygdis --public --source=. --remote=origin --push
```
