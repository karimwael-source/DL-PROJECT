# Badge Configuration Helper

## How to Update Badge URLs

Your README.md currently has placeholder badges. Update them with your actual GitHub username and repository name.

## Step 1: Find Your Username and Repo

Your repository URL looks like:
```
https://github.com/USERNAME/REPOSITORY
```

For example:
- Username: `johndoe`
- Repository: `DL-PROJECT`

## Step 2: Update README.md

Open `README.md` and find the badge section at the top. Replace:
- `<username>` with your GitHub username
- `<repo>` with your repository name

### Example:

**Before:**
```markdown
[![CI/CD Pipeline](https://github.com/<username>/<repo>/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/<username>/<repo>/actions/workflows/ci-cd.yml)
```

**After:**
```markdown
[![CI/CD Pipeline](https://github.com/johndoe/DL-PROJECT/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/johndoe/DL-PROJECT/actions/workflows/ci-cd.yml)
```

## Step 3: All Badges to Update

Update these three badges in README.md:

### 1. CI/CD Pipeline Badge
```markdown
[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci-cd.yml)
```

### 2. Tests Badge
```markdown
[![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/tests.yml)
```

### 3. Docker Build Badge
```markdown
[![Docker Build](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/docker-build.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/docker-build.yml)
```

## Optional Badges

### Docker Image Size
```markdown
[![Docker Image Size](https://ghcr-badge.egpl.dev/YOUR_USERNAME/dl-project-video-summarization/size)](https://github.com/YOUR_USERNAME/YOUR_REPO/pkgs/container/dl-project-video-summarization)
```

### License Badge
```markdown
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
```

### Python Version
```markdown
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://www.python.org/)
```

### Code Style
```markdown
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

### Codecov (after setup)
```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO)
```

## Quick Replace Command (PowerShell)

If your username is `johndoe` and repo is `DL-PROJECT`:

```powershell
# Read the file
$content = Get-Content README.md -Raw

# Replace placeholders
$content = $content -replace '<username>', 'johndoe'
$content = $content -replace '<repo>', 'DL-PROJECT'

# Save the file
$content | Set-Content README.md
```

## Quick Replace Command (Bash/Linux/Mac)

```bash
# Replace in README.md
sed -i 's/<username>/johndoe/g' README.md
sed -i 's/<repo>/DL-PROJECT/g' README.md
```

## Verify Badges Work

After updating:

1. Push changes to GitHub:
   ```bash
   git add README.md
   git commit -m "Update CI/CD badges"
   git push origin main
   ```

2. Wait for workflows to run

3. Check README on GitHub - badges should show status

4. Badges will show:
   - 🟢 **passing** - Green badge with checkmark
   - 🔴 **failing** - Red badge with X
   - ⚪ **no status** - Grey badge (workflow hasn't run yet)

## Badge Status Meanings

| Badge Text | Color | Meaning |
|------------|-------|---------|
| passing | Green | All checks passed ✓ |
| failing | Red | Some checks failed ✗ |
| no status | Grey | Workflow hasn't run yet |
| in progress | Yellow | Workflow is running |

## Complete Badge Section Example

Here's what your README.md badges section should look like (replace with your info):

```markdown
# Keyframe Detection Project

[![CI/CD Pipeline](https://github.com/johndoe/DL-PROJECT/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/johndoe/DL-PROJECT/actions/workflows/ci-cd.yml)
[![Tests](https://github.com/johndoe/DL-PROJECT/actions/workflows/tests.yml/badge.svg)](https://github.com/johndoe/DL-PROJECT/actions/workflows/tests.yml)
[![Docker Build](https://github.com/johndoe/DL-PROJECT/actions/workflows/docker-build.yml/badge.svg)](https://github.com/johndoe/DL-PROJECT/actions/workflows/docker-build.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 🚀 **CI/CD Enabled**: Automated testing, Docker builds, and deployment pipeline configured!  
> 📦 **Docker Images**: Available on GitHub Container Registry  
> 🔧 **Quick Start**: See [CI/CD Guide](docs/CI_CD_GUIDE.md) | [Quick Reference](docs/CI_CD_QUICK_REFERENCE.md)
```

## Testing Badge URLs

Before pushing, you can test if URLs are correct:

1. Replace placeholders in URLs
2. Copy the workflow URL (the part in parentheses after badge URL)
3. Paste in browser - should show your Actions page
4. If it shows 404, check spelling of username/repo

## Common Mistakes

❌ **Wrong:** `github.com/username/repo` (missing https://)  
✅ **Correct:** `https://github.com/username/repo`

❌ **Wrong:** `github.com/Username/Repo` (wrong capitalization)  
✅ **Correct:** Use exact capitalization from your GitHub URL

❌ **Wrong:** Forgetting to update both badge URL and link URL  
✅ **Correct:** Update both (there are two URLs per badge)

## Need Help?

If badges don't show up:
1. Check that workflows have run at least once
2. Verify GitHub username and repo name are correct
3. Make sure workflows are in `.github/workflows/` directory
4. Check that GitHub Actions is enabled in repo settings

## Done!

Once updated, your badges will automatically update when workflows run! 🎉
