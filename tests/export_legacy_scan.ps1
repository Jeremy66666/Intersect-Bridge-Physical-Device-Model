param([Parameter(Mandatory=$true)][string]$ScanDirectory)
$ErrorActionPreference = 'Stop'
$scan = (Resolve-Path -LiteralPath $ScanDirectory).Path
$files = @(Get-ChildItem -LiteralPath $scan -Filter 'point_*.json' | Sort-Object Name)
if ($files.Count -eq 0) { throw 'No point JSON files found' }
$exports = @()
$combined = @()
foreach ($file in $files) {
    $p = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    if ($p.status -ne 'OK' -or $p.intensity.Count -ne 2048) { throw "Invalid spectrum: $($file.Name)" }
    $hex = foreach ($value in $p.intensity) {
        if ($value -lt 0 -or $value -gt 65535 -or $value -ne [math]::Floor($value)) { throw 'Invalid intensity' }
        '{0:X4}' -f [int]$value
    }
    $time = [DateTimeOffset]::Parse($p.completed_utc).ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss.fff')
    $line = "[$time] [Backend] Image Data " + ($hex -join ' ')
    $text = "# Reconstructed from $($file.Name); timestamp = recorded completion, local time.`r`n# Mode: $($p.scan_mode); Point: $($p.point_id); XYZ controller steps: $($p.controller_x_steps), $($p.controller_y_steps), $($p.controller_z_steps)`r`n$line`r`n"
    $exports += [pscustomobject]@{Path=(Join-Path $scan ($file.BaseName + '.txt'));Text=$text;Values=$p.intensity}
    $combined += $text
}
$merged = Join-Path $scan 'laser_log_reconstructed.txt'
foreach ($path in @($exports.Path) + @($merged)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite: $path" }
}
$encoding = New-Object System.Text.UTF8Encoding($false)
foreach ($entry in $exports) {
    [IO.File]::WriteAllText($entry.Path, $entry.Text, $encoding)
    $line = Get-Content -LiteralPath $entry.Path | Where-Object { $_ -match '\[Backend\] Image Data ' }
    $tokens = ($line -split 'Image Data ',2)[1].Trim() -split '\s+'
    if ($tokens.Count -ne $entry.Values.Count) { throw 'Round-trip count mismatch' }
    for ($i=0; $i -lt $tokens.Count; $i++) {
        if ([Convert]::ToInt32($tokens[$i],16) -ne $entry.Values[$i]) { throw 'Round-trip intensity mismatch' }
    }
}
[IO.File]::WriteAllText($merged, ($combined -join "`r`n"), $encoding)
$count = @(Get-Content -LiteralPath $merged | Where-Object { $_ -match '\[Backend\] Image Data ' }).Count
if ($count -ne $files.Count) { throw 'Combined count mismatch' }
Write-Output "Verified $($files.Count) spectra x 2048 pixels; individual TXT and combined TXT saved. JSON unchanged."
Write-Output $merged
