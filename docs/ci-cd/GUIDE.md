# CI/CD Pipeline Documentation

## Overview

This project uses **GitHub Actions** for automated Continuous Integration (CI) and Continuous Deployment (CD). The pipeline automatically runs tests, builds Docker images, and can deploy the application on code changes.

## Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

**Trigger:** Push to `main`/`develop` branches, Pull Requests

**Jobs:**

#### 🧪 Test Job
- Runs on multiple Python versions (3.9, 3.10)
- Installs system and Python dependencies
- Executes all tests with pytest
- Generates code coverage reports
- Uploads coverage to Codecov

#### 🔍 Lint Job
- Checks code quality with flake8
- Validates code formatting with black
- Verifies import sorting with isort

#### 🐳 Docker Build Job
- Builds Docker image using Docker Buildx
- Pushes to GitHub Container Registry (ghcr.io)
- Tags images with branch name, SHA, and 'latest'
- Caches layers for faster builds
- Runs security scan with Docker Scout

#### 📦 Docker Hub Push (Optional)
- Pushes to Docker Hub if credentials are configured
- Only runs on main branch

#### 🚀 Deploy Job (Placeholder)
- Ready for deployment configuration
- Can be customized for your infrastructure

### 2. Docker Build Workflow (`.github/workflows/docker-build.yml`)

**Trigger:** Manual dispatch

**Purpose:** Build Docker image on-demand without running tests

**Features:**
- Custom tag input
- Saves image as artifact
- Useful for testing Docker builds

### 3. Test Suite Workflow (`.github/workflows/tests.yml`)

**Trigger:** Push, PRs, Daily schedule (2 AM UTC), Manual

**Purpose:** Comprehensive testing across platforms

**Features:**
- Matrix testing (Ubuntu, Windows × Python 3.9, 3.10)
- Parallel test execution with pytest-xdist
- Integration test support
- Test result artifacts

## Setup Instructions

### 1. Repository Setup

1. **Enable GitHub Actions:**
   - Go to your repository → Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

2. **Configure Secrets (if using Docker Hub):**
   - Go to Settings → Secrets and variables → Actions
   - Add secrets:
     - `DOCKERHUB_USERNAME`: Your Docker Hub username
     - `DOCKERHUB_TOKEN`: Docker Hub access token

### 2. Container Registry Setup

#### GitHub Container Registry (Default)

The workflow automatically pushes to GitHub Container Registry (ghcr.io):
- Image URL: `ghcr.io/<your-username>/dl-project-video-summarization`
- Authentication uses `GITHUB_TOKEN` (automatic)
- Make package public: Go to package settings → Change visibility

#### Docker Hub (Optional)

To enable Docker Hub push:
1. Create access token on Docker Hub
2. Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets
3. The workflow will automatically detect and use them

### 3. Badge Setup

Add status badges to your README.md:

```markdown
![CI/CD Pipeline](https://github.com/<username>/<repo>/actions/workflows/ci-cd.yml/badge.svg)
![Tests](https://github.com/<username>/<repo>/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/<username>/<repo>/branch/main/graph/badge.svg)](https://codecov.io/gh/<username>/<repo>)
```

## Using the Pipeline

### Automatic Triggers

1. **Push to main/develop:**
   - Runs all tests
   - Builds and pushes Docker image
   - Runs deployment (if configured)

2. **Pull Request:**
   - Runs tests on multiple platforms
   - Performs code quality checks
   - Builds Docker image (doesn't push)
   - Runs security scan

3. **Daily Schedule:**
   - Runs full test suite
   - Helps catch issues early

### Manual Triggers

1. **Run CI/CD manually:**
   ```
   Go to Actions → CI/CD Pipeline → Run workflow
   ```

2. **Build Docker image only:**
   ```
   Go to Actions → Docker Build Only → Run workflow
   Enter custom tag (optional)
   ```

3. **Run tests only:**
   ```
   Go to Actions → Test Suite → Run workflow
   ```

## Docker Image Tags

Images are automatically tagged with:
- `latest` - Latest build from main branch
- `main` - Latest build from main branch
- `develop` - Latest build from develop branch
- `main-<sha>` - Specific commit from main
- `pr-<number>` - Pull request builds
- Semantic versions (if you create release tags)

## Pull Docker Images

### From GitHub Container Registry:

```bash
# Pull latest image
docker pull ghcr.io/<username>/dl-project-video-summarization:latest

# Pull specific version
docker pull ghcr.io/<username>/dl-project-video-summarization:main-abc1234

# Run the container
docker run -p 5000:5000 ghcr.io/<username>/dl-project-video-summarization:latest
```

### From Docker Hub (if configured):

```bash
docker pull <dockerhub-username>/dl-project-video-summarization:latest
docker run -p 5000:5000 <dockerhub-username>/dl-project-video-summarization:latest
```

## Local Testing

### Run tests locally:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=src
```

### Build Docker locally:

```bash
# Build image
docker build -t dl-project-video-summarization:local .

# Run container
docker run -p 5000:5000 dl-project-video-summarization:local
```

### Test Docker Compose:

```bash
docker-compose up --build
```

## Customization

### Add Deployment

Edit `.github/workflows/ci-cd.yml` deploy job:

**For Kubernetes:**
```yaml
- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/video-summarization \
      app=ghcr.io/${{ github.repository_owner }}/dl-project-video-summarization:${{ github.sha }}
```

**For AWS ECS:**
```yaml
- name: Deploy to ECS
  uses: aws-actions/amazon-ecs-deploy-task-definition@v1
  with:
    task-definition: task-definition.json
    service: video-summarization-service
    cluster: production-cluster
```

**For Azure Container Apps:**
```yaml
- name: Deploy to Azure
  uses: azure/container-apps-deploy-action@v1
  with:
    containerAppName: video-summarization
    resourceGroup: production-rg
    imageToDeploy: ghcr.io/${{ github.repository_owner }}/dl-project-video-summarization:${{ github.sha }}
```

### Add Notifications

Add Slack/Discord notifications:

```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  if: always()
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Modify Test Configuration

Edit test job in workflows to:
- Add more Python versions
- Include additional OS (macOS)
- Add test timeouts
- Configure parallel execution

## Troubleshooting

### Tests Failing

1. Check logs in Actions tab
2. Run tests locally: `pytest tests/ -v`
3. Check Python version compatibility
4. Verify dependencies in requirements.txt

### Docker Build Failing

1. Test Dockerfile locally: `docker build -t test .`
2. Check build logs in Actions
3. Verify base image availability
4. Check disk space on runner

### Push to Registry Failing

1. Verify GITHUB_TOKEN permissions
2. Check package visibility settings
3. For Docker Hub, verify secrets are set
4. Check registry authentication

### Cache Issues

Clear GitHub Actions cache:
```bash
# Using GitHub CLI
gh cache delete <cache-id>

# Or manually through Settings → Actions → Caches
```

## Best Practices

1. **Always run tests before pushing:**
   ```bash
   pytest tests/ -v
   ```

2. **Use feature branches:**
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Write tests for new features:**
   - Add tests to `tests/` directory
   - Aim for >80% code coverage

4. **Review PR checks before merging:**
   - All tests must pass
   - Code quality checks should pass
   - Docker build must succeed

5. **Use semantic versioning for releases:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

## Monitoring

### View Workflow Runs
- Go to repository → Actions tab
- Click on specific workflow or run
- View logs, artifacts, and results

### Coverage Reports
- Uploaded to Codecov automatically
- View at: `https://codecov.io/gh/<username>/<repo>`

### Docker Images
- GitHub: `https://github.com/<username>/<repo>/pkgs/container/dl-project-video-summarization`
- Docker Hub: `https://hub.docker.com/r/<username>/dl-project-video-summarization`

## Security

The pipeline includes:
- Docker Scout vulnerability scanning
- Dependency scanning (can add Dependabot)
- Code quality checks
- Automated security updates

To enable Dependabot:
1. Create `.github/dependabot.yml`
2. Configure update schedule
3. Auto-merge low-risk updates

## Cost Optimization

GitHub Actions free tier includes:
- 2,000 minutes/month for private repos
- Unlimited for public repos

Tips:
- Use build caching (already configured)
- Run expensive tests on schedule only
- Use matrix strategy wisely
- Cancel redundant runs automatically

## Support

For issues with:
- **Workflows:** Check GitHub Actions documentation
- **Docker:** Review Docker documentation
- **Tests:** Check pytest documentation
- **Project specific:** Open an issue in this repository

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Pytest Documentation](https://docs.pytest.org/)
