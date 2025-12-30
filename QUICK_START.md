#  Quick Start Guide

## Prerequisites
- Python 3.9 or 3.10
- Docker (optional)
- Git

## 1. Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd DL-PROJECT

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Web Application

```bash
# Simple method
python webapp/app.py

# Or use the launcher
python run_webapp.py

# Visit: http://localhost:5000
```

## 3. Run with Docker

```bash
# Build and run
docker-compose up --build

# Or use helper script
.\scripts\docker_helper.ps1
```

## 4. CI/CD Setup

See [docs/ci-cd/README.md](docs/ci-cd/README.md) for complete CI/CD setup.

Quick steps:
1. Update badges in README.md
2. Push to GitHub
3. Enable GitHub Actions
4. Done!

## 5. Documentation

- **CI/CD**: [docs/ci-cd/](docs/ci-cd/)
- **Deployment**: [docs/deployment/](docs/deployment/)
- **Project Info**: [docs/project/](docs/project/)
- **All Docs**: [docs/README.md](docs/README.md)

## Need Help?

- Quick commands: [docs/ci-cd/QUICK_REFERENCE.md](docs/ci-cd/QUICK_REFERENCE.md)
- Full guide: [docs/ci-cd/GUIDE.md](docs/ci-cd/GUIDE.md)
- Issues: Open a GitHub issue
