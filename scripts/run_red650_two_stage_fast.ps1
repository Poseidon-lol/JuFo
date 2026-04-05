param(
    [string]$Input = "data/processed/opv_db_red_gap_650_780.csv",
    [string]$Shortlist = "data/processed/opv_db_red_gap_650_780_stageA_top3000.csv",
    [string]$Output = "data/processed/opv_db_red_gap_650_780_stageA_top3000_f_osc_labeled.csv",
    [int]$TopK = 3000,
    [double]$PrefilterFactor = 3.0,
    [int]$Workers = 6,
    [string]$QcConfig = "configs/qc_pipeline_f_osc_fast_win.yaml",
    [switch]$Overwrite,
    [switch]$NoResume,
    [switch]$DisableDiversity
)

$stageA = @(
    "scripts/build_red650_stageA_shortlist.py",
    "--input", $Input,
    "--output", $Shortlist,
    "--top-k", "$TopK",
    "--prefilter-factor", "$PrefilterFactor"
)
if ($DisableDiversity) {
    $stageA += "--disable-diversity"
}

Write-Host "Stage A (cheap funnel):" ($stageA -join " ")
py -3 @stageA
if ($LASTEXITCODE -ne 0) {
    throw "Stage A failed with exit code $LASTEXITCODE"
}

$stageB = @(
    "scripts/label_monomer_dataset.py",
    "--input", $Shortlist,
    "--output", $Output,
    "--qc-config", $QcConfig,
    "--workers", "$Workers",
    "--auto-multiplicity",
    "--log-level", "INFO"
)
if (-not $NoResume) {
    $stageB += "--resume"
}
if ($Overwrite) {
    $stageB += "--overwrite"
}

Write-Host "Stage B (ORCA on shortlist):" ($stageB -join " ")
py -3 @stageB
