param(
    [string]$Input = "data/processed/opv_optical_labeled_10k.csv",
    [string]$Output = "data/processed/opv_optical_labeled_10k_filtered.csv",
    [string]$FilterConfig = "configs/filter_rules_optical.yaml"
)

$argsList = @(
    "scripts/filter_qc_results.py",
    "--qc", $Input,
    "--filter", $FilterConfig,
    "--output", $Output
)

Write-Host "Running optical filter with args:" ($argsList -join " ")
py -3 @argsList
