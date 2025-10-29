#!/bin/bash
# Auto-deploy helper: commit & push to GitHub (requires git config and remote set)
git add .
git commit -m "Auto deploy from AIIS-WH2 template"
git push origin main
echo "Pushed to origin/main. For Replit, import the GitHub repo or use Replit CLI."
