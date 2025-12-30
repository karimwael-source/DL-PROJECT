# Docker Helper Script for Windows
# Quick commands for working with the Docker setup

Write-Host "=== Docker Helper Script ===" -ForegroundColor Cyan
Write-Host ""

# Build Docker image locally
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t dl-project-local:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Docker image built successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Show available commands
    Write-Host "Available Commands:" -ForegroundColor Cyan
    Write-Host "==================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "1. Run the container:" -ForegroundColor Yellow
    Write-Host "   docker run -p 5000:5000 dl-project-local:latest" -ForegroundColor White
    Write-Host ""
    
    Write-Host "2. Run with volume mount (for development):" -ForegroundColor Yellow
    Write-Host "   docker run -p 5000:5000 -v ${PWD}:/app dl-project-local:latest" -ForegroundColor White
    Write-Host ""
    
    Write-Host "3. Pull from GitHub Container Registry:" -ForegroundColor Yellow
    Write-Host "   docker pull ghcr.io/<username>/dl-project-video-summarization:latest" -ForegroundColor White
    Write-Host ""
    
    Write-Host "4. Run with docker-compose:" -ForegroundColor Yellow
    Write-Host "   docker-compose up --build" -ForegroundColor White
    Write-Host ""
    
    Write-Host "5. View running containers:" -ForegroundColor Yellow
    Write-Host "   docker ps" -ForegroundColor White
    Write-Host ""
    
    Write-Host "6. Stop containers:" -ForegroundColor Yellow
    Write-Host "   docker-compose down" -ForegroundColor White
    Write-Host ""
    
    Write-Host "7. View logs:" -ForegroundColor Yellow
    Write-Host "   docker logs <container-id>" -ForegroundColor White
    Write-Host ""
    
    Write-Host "8. Clean up:" -ForegroundColor Yellow
    Write-Host "   docker system prune -a" -ForegroundColor White
    Write-Host ""
    
    # Ask if user wants to run the container
    Write-Host "Would you like to run the container now? (y/n): " -ForegroundColor Cyan -NoNewline
    $response = Read-Host
    
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Starting container..." -ForegroundColor Yellow
        docker run -p 5000:5000 dl-project-local:latest
    }
} else {
    Write-Host "✗ Docker build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Red
}
