$BaseUrl = "https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/"
$DataDir = Join-Path $PSScriptRoot "Data"
$TinyDir = Join-Path $DataDir "tinyshakespeare"

if (!(Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }
if (!(Test-Path $TinyDir)) { New-Item -ItemType Directory -Path $TinyDir | Out-Null }

$Files = @(
    "gpt2_124M.bin",
    "gpt2_124M_bf16.bin",
    "gpt2_124M_debug_state.bin",
    "gpt2_tokenizer.bin",
    "tiny_shakespeare_train.bin",
    "tiny_shakespeare_val.bin"
)

foreach ($File in $Files) {
    $Url = "${BaseUrl}${File}?download=true"
    if ($File -like "tiny_shakespeare*") {
        $Dest = Join-Path $TinyDir $File
    } else {
        $Dest = Join-Path $DataDir $File
    }
    
    if (!(Test-Path $Dest)) {
        Write-Host "Downloading $File to $Dest..."
        Invoke-WebRequest -Uri $Url -OutFile $Dest
        Write-Host "Downloaded."
    } else {
        Write-Host "$File already exists."
    }
}
