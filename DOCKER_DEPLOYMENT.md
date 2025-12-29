# 🐳 Docker Deployment Guide for Schizego

## Prerequisites
- Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
- Make sure Docker is running

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Option 2: Using Docker Commands
```bash
# Build the image
docker build -t schizego:latest .

# Run the container
docker run -d -p 5000:5000 --name schizego-app schizego:latest

# View logs
docker logs -f schizego-app

# Stop the container
docker stop schizego-app

# Remove the container
docker rm schizego-app
```

## Access the Application
Once running, open your browser and go to:
- **http://localhost:5000**

## Deploy to Cloud

### Deploy to Railway.app
1. Push your code to GitHub (already done!)
2. Go to [railway.app](https://railway.app)
3. Click "Start a New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `DL-PROJECT` repository
6. Railway will automatically detect the Dockerfile and deploy!

### Deploy to Render.com
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect Docker
5. Click "Create Web Service"

### Deploy to Fly.io
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch
fly deploy
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs schizego-app

# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

### Port already in use
```bash
# Change port in docker-compose.yml from "5000:5000" to "8080:5000"
# Then access at http://localhost:8080
```

## Notes
- Models will be downloaded on first use (Atlas: ~100MB, Nova: ~30MB)
- Uploads are persisted via Docker volumes
- For production, consider using Gunicorn (already configured in requirements.txt)
