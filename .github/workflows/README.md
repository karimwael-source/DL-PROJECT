# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the project.

## Available Workflows

### 1. ci-cd.yml
Main CI/CD pipeline that runs on every push and PR:
- Runs tests on multiple Python versions
- Performs code quality checks
- Builds and pushes Docker images
- Ready for deployment configuration

### 2. docker-build.yml
Manual Docker build workflow:
- Build Docker image on-demand
- Test without running full CI pipeline
- Save image as artifact

### 3. tests.yml
Comprehensive test suite:
- Multi-platform testing (Ubuntu, Windows)
- Scheduled daily runs
- Integration test support
- Code coverage reporting

## Quick Start

1. Push code to trigger workflows automatically
2. View results in the Actions tab
3. Pull Docker images from GitHub Container Registry

## Documentation

See [CI_CD_GUIDE.md](../../docs/CI_CD_GUIDE.md) for complete documentation.
