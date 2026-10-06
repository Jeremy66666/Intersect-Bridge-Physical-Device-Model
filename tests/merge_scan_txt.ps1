param([Parameter(Mandatory=$true)][string]$ScanDirectory)
$ErrorActionPreference = 'Stop'
$dir = (Resolve-Path -LiteralPath $ScanDirectory).Path
$files = @(Get-ChildItem -LiteralPath $dir -Filter 'point_*.txt' | Sort-Object Name)
if ($files.Count -eq 0) { throw 'No point TXT files' }
$blocks = foreach ($file in $files) {
    $text = [IO.File]::ReadAllText($file.FullName)
    $lines = @($text -split '\r?\n' | Where-Object {$_ -match '^Image Data '})
    if ($lines.Count -ne 1) { throw "Unexpected spectrum count: $($file.Name)" }
    $tokens = $lines[0].Substring(11).Trim() -split '\s+'
    $p = Get-Content -LiteralPath (Join-Path $dir ($file.BaseName + '.json')) -Raw | ConvertFrom-Json
    if ($tokens.Count -ne 2048 -or $p.intensity.Count -ne 2048) { throw 'Invalid pixel count' }
    for ($i=0; $i -lt 2048; $i++) {
        if ([Convert]::ToInt32($tokens[$i],16) -ne $p.intensity[$i]) { throw 'TXT/JSON mismatch' }
    }
    $text.TrimEnd() + "`r`n"
}
$target = Join-Path $dir 'laser_log_combined.txt'
if (Test-Path -LiteralPath $target) { throw "Will not overwrite $target" }
[IO.File]::WriteAllText($target, ($blocks -join "`r`n"), (New-Object System.Text.UTF8Encoding($false)))
Write-Output "Saved $($files.Count) verified spectra: $target"
