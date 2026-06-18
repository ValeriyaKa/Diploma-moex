param(
    [Parameter(Mandatory=$true)]
    [string]$Image,

    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Docker daemon..."
docker info | Out-Null

$localImage = "moex-predictor:$Tag"
$remoteImage = "${Image}:$Tag"

Write-Host "Building $localImage..."
docker build -t $localImage .

Write-Host "Tagging $remoteImage..."
docker tag $localImage $remoteImage

Write-Host "Pushing $remoteImage..."
docker push $remoteImage

Write-Host ""
Write-Host "Done."
Write-Host "Use this image in Cloud.ru Container Apps:"
Write-Host $remoteImage
