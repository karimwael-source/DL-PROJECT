# CI/CD Quick Reference Card

## 🚀 Quick Actions

### View Workflow Status
```
GitHub Repo → Actions Tab
```

### Trigger Manual Build
```
Actions → CI/CD Pipeline → Run workflow (button) → Run workflow
```

### Pull Latest Docker Image
```bash
docker pull ghcr.io/<username>/dl-project-video-summarization:latest
docker run -p 5000:5000 ghcr.io/<username>/dl-project-video-summarization:latest
```

### Check Test Results
```
Actions → Latest workflow run → Test job → View logs
```

## 📋 Workflows Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI/CD Pipeline** | Push, PR | Full pipeline: test → build → deploy |
| **Tests** | Push, PR, Daily | Comprehensive testing |
| **Docker Build** | Manual | Build Docker image only |
| **Multi-Platform** | Release, Manual | Build for multiple architectures |

## 🔧 Common Commands

### Local Testing
```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html

# Code quality
flake8 src/
black --check src/
```

### Local Docker
```bash
# Build
docker build -t dl-project:local .

# Run
docker run -p 5000:5000 dl-project:local

# Test with docker-compose
docker-compose up --build
```

## 📦 Docker Image Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest from main branch |
| `main` | Latest from main branch |
| `develop` | Latest from develop branch |
| `main-abc123` | Specific commit SHA |
| `v1.0.0` | Semantic version release |

## 🔑 Required Secrets (Optional)

For Docker Hub push:
- `DOCKERHUB_USERNAME` - Your Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token

## ✅ Pre-Push Checklist

- [ ] Tests pass locally: `pytest tests/ -v`
- [ ] Code formatted: `black src/ webapp/ tests/`
- [ ] No linting errors: `flake8 src/`
- [ ] Imports sorted: `isort src/ webapp/ tests/`
- [ ] Docker builds: `docker build -t test .`
- [ ] Commit messages are clear

## 🐛 Troubleshooting

**Tests failing?**
```bash
pytest tests/ -v -s  # Show print statements
pytest tests/test_specific.py  # Run specific test
```

**Docker build failing?**
```bash
docker build -t test . --no-cache  # Build without cache
docker build -t test . --progress=plain  # Verbose output
```

**Permission denied (GitHub)?**
- Check Settings → Actions → General
- Verify "Read and write permissions" is enabled

**Docker push failing?**
- Verify secrets are set correctly
- Check package visibility settings

## 📊 Monitoring

**Coverage:** `https://codecov.io/gh/<username>/<repo>`

**Container Registry:** `https://github.com/<username>/<repo>/pkgs/container/dl-project-video-summarization`

**Workflow Runs:** `https://github.com/<username>/<repo>/actions`

## 🎯 Best Practices

1. Always work in feature branches
2. Write tests for new features
3. Keep PRs small and focused
4. Review CI checks before merging
5. Use semantic versioning for releases
6. Document breaking changes

## 📚 Documentation

Full guide: [CI_CD_GUIDE.md](CI_CD_GUIDE.md)

## 🆘 Need Help?

1. Check workflow logs in Actions tab
2. Review [CI_CD_GUIDE.md](CI_CD_GUIDE.md)
3. Search GitHub Actions documentation
4. Open an issue in the repository
