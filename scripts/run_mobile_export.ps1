param(
    [string]$Work = ".runtime/mobile_export_v1",
    [string]$Output = ".runtime/mobile_export_result_v1",
    [string]$ReuseModels = ""
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$WorkPath = (Resolve-Path -LiteralPath $Work).Path
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Output))
if (-not $OutputPath.StartsWith($ProjectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Output должен быть внутри проекта" }
if (Test-Path -LiteralPath $OutputPath) { throw "Каталог результатов уже существует: $OutputPath" }
if (-not (Test-Path -LiteralPath (Join-Path $WorkPath "input_checksums.json"))) { throw "Сначала запустите prepare_mobile_export.py" }
$ReusePath = $null
if ($ReuseModels) {
    $ReusePath = (Resolve-Path -LiteralPath $ReuseModels).Path
    foreach ($ModelName in @("model_fp32.tflite", "model_int8.tflite")) {
        if (-not (Test-Path -LiteralPath (Join-Path $ReusePath $ModelName))) { throw "Missing reused model: $ModelName" }
    }
}
function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Ошибка docker (exit $LASTEXITCODE): $args" }
}
$RunId = [guid]::NewGuid().ToString("N")
$Volume = "nevidal-mobile-$RunId"
$Container = "nevidal-mobile-$RunId"
# Только небольшой build-context: ни raw, ни .venv, ни .git не передаются Docker builder.
$Context = Join-Path $ProjectRoot ".runtime/mobile-docker-context-$RunId"
New-Item -ItemType Directory -Path $Context | Out-Null
Copy-Item -LiteralPath "docker/mobile-export/Dockerfile" -Destination (Join-Path $Context "Dockerfile")
Copy-Item -LiteralPath "requirements-mobile-export.txt" -Destination $Context
Invoke-Docker build -t nevidal-mobile-export:local $Context
Invoke-Docker volume create $Volume
$CreateArgs = @("create", "--name", $Container, "--mount", "type=volume,source=$Volume,target=/work", "nevidal-mobile-export:local")
if ($ReusePath) {
    $CreateArgs += @("python", "scripts/export_mobile_litert.py", "--work", "/work/input", "--output", "/work/output", "--reuse-models", "/work/previous")
}
Invoke-Docker @CreateArgs
# docker cp работает без проблемного bind mount I:\Невидаль.
Invoke-Docker cp "${WorkPath}" "${Container}:/work/input"
Invoke-Docker cp "src" "${Container}:/app/src"
Invoke-Docker cp "scripts" "${Container}:/app/scripts"
if ($ReusePath) { Invoke-Docker cp "${ReusePath}" "${Container}:/work/previous" }
& docker start --attach $Container
# При ошибке Python всё равно копируем status.json и traceback на хост.
$ExportExit = & docker inspect --format '{{.State.ExitCode}}' $Container
if ($LASTEXITCODE -ne 0) { throw "Не удалось прочитать статус контейнера" }
New-Item -ItemType Directory -Path $OutputPath | Out-Null
Invoke-Docker cp "${Container}:/work/output/." $OutputPath
Invoke-Docker cp "${Container}:/opt/pip-freeze.txt" (Join-Path $OutputPath "pip-freeze.txt")
& docker image inspect nevidal-mobile-export:local --format '{{.Id}}' | Set-Content -LiteralPath (Join-Path $OutputPath "docker-image-id.txt")
Write-Host "Результаты: $OutputPath"
Write-Host "Контейнер и volume сохранены для диагностики: $Container / $Volume"
if ([int]$ExportExit -ne 0) { throw "Экспорт завершился ошибкой. Смотрите status.json" }
