# Script to copy data files to the data/raw directory
$sourceDir = "C:\Users\nnsje\OneDrive\Desktop\gene"
$destDir = "C:\Users\nnsje\OneDrive\Desktop\gene\data\raw"

# List of files to copy
$filesToCopy = @("train.csv", "test.csv", "orthogonal.csv", "acmg_guided.csv", "denovo.csv")

# Create destination directory if it doesn't exist
if (-not (Test-Path -Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir | Out-Null
}

# Copy each file if it exists in the source directory
foreach ($file in $filesToCopy) {
    $sourcePath = Join-Path -Path $sourceDir -ChildPath $file
    $destPath = Join-Path -Path $destDir -ChildPath $file
    
    if (Test-Path -Path $sourcePath) {
        Write-Host "Copying $file to $destDir"
        Copy-Item -Path $sourcePath -Destination $destPath -Force
    } else {
        Write-Host "Warning: $file not found in $sourceDir"
    }
}

Write-Host "File copy complete!"
