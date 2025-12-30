# 🚀 CI/CD Setup Checklist

Use this checklist to ensure your CI/CD pipeline is properly configured.

## ✅ Initial Setup

### GitHub Repository
- [ ] Repository created on GitHub
- [ ] Code pushed to repository
- [ ] `.github/workflows/` directory exists
- [ ] Workflow files are present:
  - [ ] `ci-cd.yml`
  - [ ] `tests.yml`
  - [ ] `docker-build.yml`
  - [ ] `docker-multiplatform.yml`

### GitHub Actions Configuration
- [ ] Go to Settings → Actions → General
- [ ] Enable "Allow all actions and reusable workflows"
- [ ] Enable "Read and write permissions" for GITHUB_TOKEN
- [ ] Enable "Allow GitHub Actions to create and approve pull requests"

## 🐳 Docker Registry Setup

### GitHub Container Registry (Automatic)
- [ ] First workflow run completed successfully
- [ ] Package appears in your profile/organization packages
- [ ] Go to Package settings
- [ ] Change visibility to Public (if desired)
- [ ] Verify image pull works:
  ```bash
  docker pull ghcr.io/<username>/dl-project-video-summarization:latest
  ```

### Docker Hub (Optional)
- [ ] Docker Hub account created
- [ ] Create access token on Docker Hub
  - [ ] Go to Account Settings → Security → Access Tokens
  - [ ] Click "New Access Token"
  - [ ] Name it "GitHub Actions"
  - [ ] Copy the token
- [ ] Add secrets to GitHub:
  - [ ] Go to Settings → Secrets and variables → Actions
  - [ ] Add `DOCKERHUB_USERNAME` secret
  - [ ] Add `DOCKERHUB_TOKEN` secret
- [ ] Push code to trigger workflow
- [ ] Verify image on Docker Hub

## 🧪 Testing Setup

### Local Testing
- [ ] Install test dependencies:
  ```bash
  pip install pytest pytest-cov black flake8 isort
  ```
- [ ] Run tests locally:
  ```bash
  pytest tests/ -v
  ```
- [ ] Tests pass locally
- [ ] Run pre-push checker:
  ```bash
  python scripts/check_ci.py
  ```

### GitHub Actions Tests
- [ ] Push code to trigger tests
- [ ] View Actions tab
- [ ] All test jobs pass
- [ ] Coverage report generated

## 📊 Code Quality

### Configuration
- [ ] `setup.cfg` file exists
- [ ] `.gitignore` updated with CI/CD entries

### Checks
- [ ] Run black:
  ```bash
  black src/ webapp/ tests/
  ```
- [ ] Run flake8:
  ```bash
  flake8 src/ --max-line-length=127
  ```
- [ ] Run isort:
  ```bash
  isort src/ webapp/ tests/
  ```
- [ ] All checks pass

## 📝 Documentation

### Files Created
- [ ] `CI_CD_IMPLEMENTATION.md` exists
- [ ] `docs/CI_CD_GUIDE.md` exists
- [ ] `docs/CI_CD_QUICK_REFERENCE.md` exists
- [ ] `.github/workflows/README.md` exists
- [ ] `.github/PULL_REQUEST_TEMPLATE.md` exists

### README Updates
- [ ] Status badges added to README.md
- [ ] Badge URLs updated with your username/repo
- [ ] Badges link to correct workflow files

## 🔒 Security

### Dependabot
- [ ] `.github/dependabot.yml` file exists
- [ ] Dependabot alerts enabled in Settings → Security
- [ ] First dependency update PR appears

### Security Scanning
- [ ] Docker Scout scan runs in workflow
- [ ] Trivy scan enabled in multi-platform workflow
- [ ] Security tab shows scan results

## 🚀 First Deployment

### Initial Push
- [ ] Commit all CI/CD files:
  ```bash
  git add .github/ docs/ scripts/ setup.cfg .gitignore
  git commit -m "Add CI/CD pipeline with GitHub Actions"
  git push origin main
  ```

### Verify Workflow Runs
- [ ] Go to Actions tab
- [ ] "CI/CD Pipeline" workflow is running
- [ ] Watch build progress
- [ ] All jobs complete successfully (green checkmarks)

### Verify Docker Image
- [ ] Image pushed to GitHub Container Registry
- [ ] Image tagged correctly (latest, main, SHA)
- [ ] Pull image:
  ```bash
  docker pull ghcr.io/<username>/dl-project-video-summarization:latest
  ```
- [ ] Run container:
  ```bash
  docker run -p 5000:5000 ghcr.io/<username>/dl-project-video-summarization:latest
  ```
- [ ] Application works correctly

## 📈 Monitoring

### Status Badges
- [ ] Update badge URLs in README.md
- [ ] Replace `<username>` with your GitHub username
- [ ] Replace `<repo>` with your repository name
- [ ] Badges display correctly in README
- [ ] Badges show correct status

### Codecov (Optional)
- [ ] Sign up at codecov.io
- [ ] Connect your GitHub repository
- [ ] Add `CODECOV_TOKEN` secret
- [ ] Coverage reports upload successfully
- [ ] View coverage dashboard

## 🔄 Workflow Testing

### Manual Triggers
- [ ] Test CI/CD Pipeline:
  - [ ] Go to Actions → CI/CD Pipeline
  - [ ] Click "Run workflow"
  - [ ] Workflow runs successfully
  
- [ ] Test Docker Build:
  - [ ] Go to Actions → Docker Build Only
  - [ ] Click "Run workflow"
  - [ ] Enter custom tag
  - [ ] Build completes
  - [ ] Artifact is created

### Scheduled Tests
- [ ] Tests workflow scheduled (check `.github/workflows/tests.yml`)
- [ ] Wait for scheduled run (or trigger manually)
- [ ] Verify scheduled test runs

## 📋 Pull Request Testing

### Create Test PR
- [ ] Create new branch:
  ```bash
  git checkout -b test/ci-cd-verification
  ```
- [ ] Make a small change
- [ ] Push branch:
  ```bash
  git push origin test/ci-cd-verification
  ```
- [ ] Create Pull Request on GitHub

### Verify PR Checks
- [ ] PR shows checks running
- [ ] All required checks pass:
  - [ ] Test job
  - [ ] Lint job
  - [ ] Docker build job
- [ ] Status shows in PR conversation
- [ ] Can merge after checks pass

## 🎯 Advanced Features (Optional)

### Deployment
- [ ] Choose deployment target
- [ ] Configure deployment in `ci-cd.yml`
- [ ] Add deployment secrets
- [ ] Test deployment
- [ ] Verify application is accessible

### Notifications
- [ ] Set up Slack/Discord webhook
- [ ] Add webhook URL to secrets
- [ ] Add notification step to workflows
- [ ] Test notifications

### Multi-Platform Builds
- [ ] Test multi-platform workflow
- [ ] Verify ARM64 build works
- [ ] Test on different architectures

## ✨ Final Verification

### Complete System Test
- [ ] Make a code change
- [ ] Commit and push
- [ ] Workflow triggers automatically
- [ ] Tests run and pass
- [ ] Docker image builds
- [ ] Image pushes to registry
- [ ] Pull and run new image
- [ ] Application works correctly

### Documentation Review
- [ ] Read through CI_CD_GUIDE.md
- [ ] Understand workflow triggers
- [ ] Know how to pull Docker images
- [ ] Understand troubleshooting steps

## 📞 Support

If you encounter issues:
- [ ] Check workflow logs in Actions tab
- [ ] Review [CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)
- [ ] Run local checks: `python scripts/check_ci.py`
- [ ] Check [CI_CD_QUICK_REFERENCE.md](docs/CI_CD_QUICK_REFERENCE.md)
- [ ] Search GitHub Actions documentation
- [ ] Open an issue if needed

## 🎉 Success!

When all items are checked:
- ✅ Your CI/CD pipeline is fully operational
- ✅ Tests run automatically on every push
- ✅ Docker images build and deploy automatically
- ✅ Code quality is enforced
- ✅ Security scanning is active
- ✅ You're ready for production!

---

**Date Completed:** _______________

**Tested By:** _______________

**Notes:** _______________________________________________
