<#
.SYNOPSIS
    Manage Git worktrees for parallel and sequential multi-LLM execution across Waves 1–8.
.DESCRIPTION
    Creates, lists, and cleans isolated Git worktrees under .worktrees/
    while enforcing single-contributor Git authorship (Hans Darmawan <work.hansdarmawan001@gmail.com>),
    shared .githooks, and BSB_DATA_DIR=../.. environment configuration.
#>
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("init", "list", "clean")]
    [string]$Action = "list",

    [Parameter(Mandatory = $false)]
    [int]$Wave = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$CandidateRoot = Split-Path -Parent $ScriptDir
if ((Split-Path -Leaf (Split-Path -Parent $CandidateRoot)) -in @(".worktrees", "worktrees")) {
    $RepoRoot = Split-Path -Parent (Split-Path -Parent $CandidateRoot)
} else {
    $RepoRoot = $CandidateRoot
}
Set-Location $RepoRoot

$Waves = @(
    # Wave 1 (Parallel)
    @{ Wave = 1; Name = "wt-data-usj-pdf"; Branch = "feat/ingest-usj-pdf"; Task = "TASK-02"; Agent = "Claude Code" },
    @{ Wave = 1; Name = "wt-data-xlsx";    Branch = "feat/ingest-xlsx";    Task = "TASK-03"; Agent = "Codex" },
    @{ Wave = 1; Name = "wt-qdrant-infra"; Branch = "feat/qdrant-infra";   Task = "TASK-04"; Agent = "Antigravity" },

    # Wave 2 (Parallel)
    @{ Wave = 2; Name = "wt-retrieval";    Branch = "feat/retrieval-pipeline"; Task = "TASK-05"; Agent = "Claude Code" },
    @{ Wave = 2; Name = "wt-llm-app";      Branch = "feat/llm-app";            Task = "TASK-06"; Agent = "Codex" },

    # Wave 3 (Sequential Integration)
    @{ Wave = 3; Name = "wt-integration";  Branch = "feat/integration-eval-docs"; Task = "TASK-07"; Agent = "Antigravity" },

    # Wave 4 (Sequential Prerequisite)
    @{ Wave = 4; Name = "wt-p0-correctness"; Branch = "fix/p0-correctness"; Task = "TASK-08"; Agent = "Codex" },

    # Wave 5 (Sequential)
    @{ Wave = 5; Name = "wt-offline-llm";    Branch = "feat/offline-llm";   Task = "TASK-09"; Agent = "Claude Code" },

    # Wave 6 (Parallel)
    @{ Wave = 6; Name = "wt-streamlit-ui";   Branch = "feat/streamlit-ui";  Task = "TASK-10"; Agent = "Codex" },
    @{ Wave = 6; Name = "wt-quality-eval";   Branch = "feat/quality-eval";  Task = "TASK-11"; Agent = "Claude Code" },

    # Wave 7 (Sequential)
    @{ Wave = 7; Name = "wt-ops-hardening";  Branch = "feat/ops-hardening"; Task = "TASK-12"; Agent = "Codex" },

    # Wave 8 (Final Sequential)
    @{ Wave = 8; Name = "wt-repo-hygiene";   Branch = "chore/repo-hygiene"; Task = "TASK-13"; Agent = "Antigravity" }
)

function Set-GitAuthorHygiene {
    param([string]$TargetDir)
    Push-Location $TargetDir
    try {
        git config user.name "Hans Darmawan"
        git config user.email "work.hansdarmawan001@gmail.com"
        git config core.hooksPath ".githooks"
    } finally {
        Pop-Location
    }
}

function Configure-WorktreeEnv {
    param([string]$TargetDir)
    $envPath = Join-Path $TargetDir ".env"
    $envExample = Join-Path $RepoRoot ".env.example"

    if (-not (Test-Path $envPath)) {
        if (Test-Path $envExample) {
            Copy-Item $envExample $envPath
        } else {
            New-Item -ItemType File -Path $envPath | Out-Null
        }
    }

    $content = Get-Content $envPath -Raw -ErrorAction SilentlyContinue
    if ($null -eq $content) { $content = "" }

    if ($content -match "BSB_DATA_DIR=") {
        $content = $content -replace "BSB_DATA_DIR=.*", "BSB_DATA_DIR=../.."
    } else {
        $content = "BSB_DATA_DIR=../../`r`n" + $content
    }
    Set-Content -Path $envPath -Value $content -NoNewline
}

function Initialize-WorktreeSet {
    param([array]$TargetWorktrees)
    Set-GitAuthorHygiene -TargetDir $RepoRoot
    $WtBase = Join-Path $RepoRoot ".worktrees"
    if (-not (Test-Path $WtBase)) {
        New-Item -ItemType Directory -Path $WtBase | Out-Null
    }

    foreach ($wt in $TargetWorktrees) {
        $wtPath = Join-Path $WtBase $wt.Name
        if (Test-Path $wtPath) {
            Write-Host "[SKIP] Worktree $($wt.Name) already exists at $wtPath ($($wt.Task) - $($wt.Agent))" -ForegroundColor Yellow
            Set-GitAuthorHygiene -TargetDir $wtPath
            Configure-WorktreeEnv -TargetDir $wtPath
            continue
        }

        $localBranch = git branch --list $wt.Branch
        $remoteBranch = git branch -r --list "origin/$($wt.Branch)"

        if ($localBranch) {
            git worktree add $wtPath $wt.Branch
        } elseif ($remoteBranch) {
            git worktree add --track -b $wt.Branch $wtPath "origin/$($wt.Branch)"
        } else {
            git worktree add -b $wt.Branch $wtPath origin/main
        }

        Set-GitAuthorHygiene -TargetDir $wtPath
        Configure-WorktreeEnv -TargetDir $wtPath
        Write-Host "[OK] Created worktree $($wt.Name) on branch $($wt.Branch) -> $($wt.Task) ($($wt.Agent))" -ForegroundColor Green
    }
}

switch ($Action) {
    "init" {
        if ($Wave -gt 0) {
            $selected = $Waves | Where-Object { $_.Wave -eq $Wave }
            if (-not $selected) {
                Write-Error "No worktrees defined for Wave $Wave."
                return
            }
            Initialize-WorktreeSet -TargetWorktrees $selected
        } else {
            Initialize-WorktreeSet -TargetWorktrees $Waves
        }
    }

    "list" {
        git worktree list
    }

    "clean" {
        Set-GitAuthorHygiene -TargetDir $RepoRoot
        git fetch origin --prune

        $mergedRemote = (git branch -r --merged origin/main) | ForEach-Object { $_.Trim().Replace("origin/", "") }
        $mergedLocal = (git branch --merged origin/main) | ForEach-Object { $_.Trim().Replace("* ", "") }
        $mergedSet = ($mergedRemote + $mergedLocal) | Select-Object -Unique

        $WtBase = Join-Path $RepoRoot ".worktrees"
        foreach ($wt in $Waves) {
            $wtPath = Join-Path $WtBase $wt.Name
            if (Test-Path $wtPath) {
                $isMerged = $mergedSet -contains $wt.Branch
                if (-not $isMerged) {
                    git merge-base --is-ancestor $wt.Branch origin/main 2>$null
                    if ($LASTEXITCODE -eq 0) {
                        $isMerged = $true
                    }
                }
                if ($isMerged) {
                    Write-Host "Removing merged worktree $($wt.Name) at $wtPath ($($wt.Branch))..." -ForegroundColor Yellow
                    git worktree remove --force $wtPath
                    Write-Host "[REMOVED] $wtPath" -ForegroundColor Green
                } else {
                    Write-Host "[KEEP] Worktree $($wt.Name) ($($wt.Branch)) is not yet merged into origin/main." -ForegroundColor Cyan
                }
            }
        }
        git worktree prune
    }
}
