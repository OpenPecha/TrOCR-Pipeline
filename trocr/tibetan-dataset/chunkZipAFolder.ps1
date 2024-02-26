# Load required assembly for runspaces
Add-Type -AssemblyName System.Threading

# Define the source folder and the target folder for the zip files
$sourceFolder = "C:\Users\301212298\PyCharmProjects\TrOcr-Tibetan-Fine-tuning\trocr\tibetan-dataset\train"
$targetFolder = "C:\Users\301212298\PyCharmProjects\TrOcr-Tibetan-Fine-tuning\trocr\tibetan-dataset"

# Ensure the target folder exists
if (-not (Test-Path -Path $targetFolder)) {
	Write-Host 'Folder not found'
    exit
}

# Retrieve all files in the source directory
$files = Get-ChildItem -Path $sourceFolder -File
$filesPerZip = [Math]::Ceiling($files.Count / 10)

# Define a script block for parallel compression
$scriptBlock = {
    param($filesToZip, $zipFilePath)
    Compress-Archive -Path $filesToZip.FullName -DestinationPath $zipFilePath
}

# Create and configure RunspacePool for parallel execution
$runspacePool = [runspacefactory]::CreateRunspacePool(1, [Environment]::ProcessorCount)
$runspacePool.Open()

$runspaces = @()

for ($i = 0; $i -lt 10; $i++) {
    $subset = $files | Select-Object -Skip ($i * $filesPerZip) -First $filesPerZip
    $zipFileName = "Archive_Part_$($i+1).zip"
    $zipFilePath = Join-Path -Path $targetFolder -ChildPath $zipFileName

    # Create a PowerShell instance and add the script block with arguments
    $powershell = [powershell]::Create().AddScript($scriptBlock).AddArgument($subset).AddArgument($zipFilePath)
    $powershell.RunspacePool = $runspacePool

    # Begin the asynchronous execution and store the handle
    $runspaces += New-Object PSObject -Property @{
        Runspace = $powershell.BeginInvoke()
        PowerShell = $powershell
    }
}

# Wait for all runspaces to complete
$runspaces | ForEach-Object {
    $_.PowerShell.EndInvoke($_.Runspace)
    $_.PowerShell.Dispose()
}

$runspacePool.Close()
$runspacePool.Dispose()

Write-Output "All files have been divided into 10 zip files in parallel."