param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [string]$Output = "merged.mp4",

    # 強制轉檔再合併（避免相容性問題）
    [switch]$Transcode,

    # 轉檔時保留中間檔（預設轉完會清掉）
    [switch]$KeepTemp,

    # 合併排序依據
    [ValidateSet('Path','Name','CreationTime','LastWriteTime')]
    [string]$SortBy = 'Path',

    # 要納入的副檔名（可自行增減）
    [string[]]$Extensions = @('*.mp4','*.mov','*.mkv','*.m4v','*.avi','*.ts','*.webm'),

    # 轉檔失敗時跳過該檔
    [switch]$ContinueOnError,

    # 指定中間檔暫存資料夾（例如 F:\fftmp）
    [string]$WorkDir
)

function Assert-FFmpeg {
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        throw "找不到 ffmpeg。請先安裝 FFmpeg 並確保 ffmpeg 在 PATH 中可被呼叫。"
    }
}

function Resolve-Folder($p) {
    $rp = Resolve-Path -Path $p -ErrorAction Stop
    $item = Get-Item -LiteralPath $rp -ErrorAction Stop
    if (-not $item.PSIsContainer) { throw "提供的 Path 不是資料夾：$p" }
    return $item.FullName
}

function Get-VideoFiles($root, $exts) {
    $files = @()
    foreach ($pat in $exts) {
        $files += Get-ChildItem -LiteralPath $root -Recurse -File -Include $pat -ErrorAction SilentlyContinue
    }
    if (-not $files -or $files.Count -eq 0) {
        throw "在 `$root` 找不到符合副檔名的影片檔。"
    }
    return $files | Sort-Object -Property FullName -Unique
}

function Sort-Files($files, $key) {
    switch ($key) {
        'Path'          { return $files | Sort-Object FullName }
        'Name'          { return $files | Sort-Object Name, DirectoryName }
        'CreationTime'  { return $files | Sort-Object CreationTime, FullName }
        'LastWriteTime' { return $files | Sort-Object LastWriteTime, FullName }
        default         { return $files | Sort-Object FullName }
    }
}

function Invoke-FFmpeg($argsArray) {
    & ffmpeg @argsArray
    return $LASTEXITCODE
}

# ✅ 支援 FileInfo 或 string，並用 UTF-8 無 BOM 輸出清單
function Write-ListFile($files, $listPath) {
    $lines = foreach ($f in $files) {
        $p = if ($f -is [string]) { $f } else { $f.FullName }
        $p = ($p -replace '\\','/')
        # concat 規格：單引號以 '\'' 跳脫
        $p = $p -replace "'", "'\''"
        "file '$p'"
    }

    if ($PSVersionTable.PSVersion.Major -ge 6) {
        Set-Content -Path $listPath -Value $lines -Encoding utf8NoBOM
    } else {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllLines($listPath, $lines, $utf8NoBom)
    }
}

# ✅ 若指定 WorkDir，就把清單檔寫到那裡，避免用到 C:\TEMP
function Try-ConcatCopy($files, $outPath, $workDir) {
    $baseTemp = if ($workDir) { $workDir } else { [System.IO.Path]::GetTempPath() }
    if (-not (Test-Path $baseTemp)) { New-Item -ItemType Directory -Path $baseTemp | Out-Null }
    $tmp = Join-Path $baseTemp ("fflist_" + [Guid]::NewGuid().ToString() + ".txt")
    try {
        Write-ListFile -files $files -listPath $tmp
        $args = @(
            "-hide_banner",
            "-f","concat","-safe","0","-i",$tmp,
            "-c","copy",
            "-y",$outPath
        )
        $code = Invoke-FFmpeg $args
        return $code -eq 0
    } finally {
        if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
    }
}

# ✅ 逐檔 try/catch，失敗依參數選擇略過或中止；暫存路徑使用 WorkDir
function Concat-ByTranscode($files, $outPath, $keepTemp, $continueOnError, $workDir) {
    $baseTemp = if ($workDir) { $workDir } else { [System.IO.Path]::GetTempPath() }
    if (-not (Test-Path $baseTemp)) { New-Item -ItemType Directory -Path $baseTemp | Out-Null }
    $tempDir = Join-Path $baseTemp ("ffmerge_" + [Guid]::NewGuid().ToString())
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    $okList   = New-Object System.Collections.Generic.List[string]
    $failList = New-Object System.Collections.Generic.List[string]

    try {
        for ($i = 0; $i -lt $files.Count; $i++) {
            $in  = $files[$i].FullName
            $out = Join-Path $tempDir ("{0:D5}.mp4" -f $i)

            Write-Host "轉檔中 [$($i+1)/$($files.Count)]: $in"
            $args = @(
                "-hide_banner",
                "-v","error",            # 要除錯可改 warning 或移除
                "-i",$in,
                "-map","0:v:0?","-map","0:a:0?",
                "-c:v","libx264","-preset","medium","-crf","20",
                "-pix_fmt","yuv420p",
                "-c:a","aac","-b:a","192k","-ar","48000",
                "-movflags","+faststart",
                "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-y",$out
            )

            $code = Invoke-FFmpeg $args

            if ($code -ne 0) {
                if (Test-Path $out) { Remove-Item $out -Force -ErrorAction SilentlyContinue }
                if ($continueOnError) {
                    Write-Warning "轉檔失敗，已略過：$in"
                    $failList.Add($in) | Out-Null
                    continue
                } else {
                    throw "轉檔失敗：$in"
                }
            }

            $okList.Add($out) | Out-Null
        }

        if ($okList.Count -eq 0) {
            throw "沒有成功轉檔的影片可供合併。"
        }

        # 建清單並合併
        $listFile = Join-Path $tempDir "list.txt"
        Write-ListFile -files $okList -listPath $listFile

        Write-Host "合併中 ..."
        $mergeArgs = @(
            "-hide_banner",
            "-v","error",
            "-f","concat","-safe","0","-i",$listFile,
            "-c","copy",
            "-y",$outPath
        )
        $mergeCode = Invoke-FFmpeg $mergeArgs
        if ($mergeCode -ne 0) { throw "合併失敗（轉檔後）。" }

        if ($failList.Count -gt 0) {
            Write-Warning ("共有 {0} 個檔案被略過。" -f $failList.Count)
            $failList | ForEach-Object { Write-Warning "略過：$_" }
        }
    }
    finally {
        if (-not $keepTemp) {
            Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "已保留中間檔於：$tempDir"
        }
    }
}

# ================== Main ==================
try {
    Assert-FFmpeg
    $root = Resolve-Folder $Path
    $files = Get-VideoFiles -root $root -exts $Extensions
    $files = Sort-Files -files $files -key $SortBy

    Write-Host "找到 $($files.Count) 個影片檔，開始處理 ..."
    $outFull = [System.IO.Path]::GetFullPath($Output)
    $outDir  = Split-Path -Parent $outFull
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

    if ($files.Count -eq 1) {
        Write-Host "只有一支影片，直接複製輸出（不重編碼）..."
        $copyArgs = @("-hide_banner","-i",$files[0].FullName,"-c","copy","-y",$outFull)
        $code = Invoke-FFmpeg $copyArgs
        if ($code -ne 0) { throw "輸出失敗。" }
        Write-Host "完成：$outFull"
        exit 0
    }

    if ($Transcode) {
        Concat-ByTranscode -files $files -outPath $outFull -keepTemp:$KeepTemp `
                           -continueOnError:$ContinueOnError -workDir $WorkDir
    } else {
        Write-Host "嘗試無重編碼串接 ..."
        $ok = Try-ConcatCopy -files $files -outPath $outFull -workDir $WorkDir
        if (-not $ok) {
            Write-Warning "無重編碼串接失敗，改用轉檔再合併（較花時間但最穩）。"
            Concat-ByTranscode -files $files -outPath $outFull -keepTemp:$KeepTemp `
                               -continueOnError:$ContinueOnError -workDir $WorkDir
        }
    }

    Write-Host "✅ 完成輸出：$outFull"
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}


# D:\darknet\lilin-tool\merge_videos.ps1 `
#   -Path   "F:\another\20250527_現場錄製訓練用影片\short" `
#   -Output "F:\another\20250527_現場錄製訓練用影片\merge.mp4" `
#   -ContinueOnError -WorkDir "F:\fftmp"