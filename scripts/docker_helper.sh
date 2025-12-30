# Docker Helper Script
# Quick commands for working with the Docker setup

# Build Docker image locally
echo "Building Docker image..."
docker build -t dl-project-local:latest .

# Run the container
echo ""
echo "To run the container:"
echo "docker run -p 5000:5000 dl-project-local:latest"
echo ""

# Pull from GitHub Container Registry
echo "To pull from GitHub Container Registry (after CI/CD runs):"
echo "docker pull ghcr.io/<username>/dl-project-video-summarization:latest"
echo ""

# Test with docker-compose
echo "To test with docker-compose:"
echo "docker-compose up --build"
echo ""

# View running containers
echo "To view running containers:"
echo "docker ps"
echo ""

# Stop all containers
echo "To stop all containers:"
echo "docker-compose down"
echo ""

# Clean up
echo "To clean up unused Docker resources:"
echo "docker system prune -a"
