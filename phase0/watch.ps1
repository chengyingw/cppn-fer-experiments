# Shows the running job and the latest log lines every 30 seconds. Ctrl-C stops the display, not the experiments.
$here = $PSScriptRoot
while ($true) {
    Clear-Host
    Write-Output ("Now " + (Get-Date -Format 'HH:mm:ss'))
    Write-Output "--- running"
    $jobs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_|eval_controls|continual' }
    if ($jobs) {
        foreach ($j in $jobs) { Write-Output (($j.CommandLine -replace '^.*python\.exe"\s*', '') + "  (since " + $j.CreationDate.ToString('HH:mm:ss') + ")") }
    } else {
        Write-Output "no training job running"
    }
    foreach ($name in 'runs_b6.log', 'runs_continual.log') {
        Write-Output "--- $name"
        $path = Join-Path $here $name
        if (Test-Path $path) {
            Get-Content $path -Tail 8 | ForEach-Object { $_.Substring(0, [Math]::Min(110, $_.Length)) }
        }
    }
    Start-Sleep -Seconds 30
}
