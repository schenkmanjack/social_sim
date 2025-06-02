# For macOS/Linux
for port in {5000..5021} 8080; do
    lsof -ti :$port | xargs kill -9 2>/dev/null || echo "No process on port $port"
done

# For Windows (PowerShell)
foreach ($port in 5000..5021 + 8080) {
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id (Get-Process -Id $process.OwningProcess).Id -Force
    }
}