param(
    [string]$From = "11.05",
    [string]$To = "today",
    [switch]$Save
)

$ErrorActionPreference = "Stop"
$Python = "C:\Diploma\dip\Scripts\python.exe"

Write-Host "1/3 Loading candles to Supabase..."
& $Python load_candles.py --from $From --till $To

Write-Host "2/3 Evaluating saved predictions..."
$argsList = @("evaluate_predictions.py", "--from", $From, "--to", $To)
if ($Save) {
    $argsList += "--save"
}
& $Python @argsList

Write-Host "3/3 Generating fresh predictions for the next trading day..."
& $Python generate_predictions.py
