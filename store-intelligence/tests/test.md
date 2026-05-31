# 1. Spin up the container services in detached state
docker compose up --build -d

# 2. Confirm container initialization health checks are green
docker compose ps

# 3. Open an execution shell to run your integration verification test suite
docker compose exec analytics-api pytest tests/ -v