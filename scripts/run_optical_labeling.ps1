param(
    [string]$Input = "data/processed/opv_optical_shortlist_10k.csv",
    [string]$Output = "data/processed/opv_optical_labeled_10k.csv",
    [string]$QcConfig = "configs/qc_pipeline_optical.yaml",
    [int]$Workers = 10,
    [switch]$Overwrite,
    [switch]$NoResume
)

$argsList = @(
    "scripts/label_monomer_dataset.py",
    "--input", $Input,
    "--output", $Output,
    "--qc-config", $QcConfig,
    "--workers", "$Workers"
)

if (-not $NoResume) {
    $argsList += "--resume"
}
if ($Overwrite) {
    $argsList += "--overwrite"
}

Write-Host "Running optical labeling with args:" ($argsList -join " ")
py -3 @argsList
