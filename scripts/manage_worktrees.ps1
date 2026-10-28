<#
.SYNOPSIS
    Manage Git worktrees for parallel multi-LLM execution (Antigravity, Claude Code, Codex).
.DESCRIPTION
    Creates, lists, merges, and cleans isolated Git worktrees under .worktrees/
    while enforcing single-contributor Git authorship (Hans Darmawan <work.hansdarmawan001@gmail.com>)
    and shared .githooks across all worktrees.
#>
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("init", "init-wave1", "init-wave2", "list", "merge-wave1", "merge-wave2", "clean")]
    [string]$Action = "list"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Wave1Worktrees = @(
    @{ Name = "wt-data-usj-pdf"; Branch = "feat/ingest-usj-pdf"; Agent = "Claude Code (TASK-02)" },
    @{ Name = "wt-data-xlsx";    Branch = "feat/ingest-xlsx";    Agent = "Codex (TASK-03)" },
    @{ Name = "wt-qdrant-infra"; Branch = "feat/qdrant-infra";   Agent = "Antigravity (TASK-04)" }
)

$Wave2Worktrees = @(
    @{ Name = "wt-retrieval";    Branch = "feat/retrieval-pipeline"; Agent = "Claude Code (TASK-05)" },
    @{ Name = "wt-llm-app";      Branch = "feat/llm-app";            Agent = "Codex (TASK-06)" }
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

function Initialize-WorktreeSet {
    param([array]$Worktrees)
    Set-GitAuthorHygiene -TargetDir $RepoRoot
    $WtBase = Join-Path $RepoRoot ".worktrees"
    if (-not (Test-Path $WtBase)) {
        New-Item -ItemType Directory -Path $WtBase | Out-Null
    }
    foreach ($wt in $Worktrees) {
        $wtPath = Join-Path $WtBase $wt.Name
        if (Test-Path $wtPath) {
            Write-Host "[SKIP] Worktree $($wt.Name) already exists at $wtPath ($($wt.Agent))" -ForegroundColor Yellow
            Set-GitAuthorHygiene -TargetDir $wtPath
            continue
        }
        $branchExists = git branch --list $wt.Branch
        if ($branchExists) {
            git worktree add $wtPath $wt.Branch
        } else {
            git worktree add -b $wt.Branch $wtPath main
        }
        Set-GitAuthorHygiene -TargetDir $wtPath
        Write-Host "[OK] Created worktree $($wt.Name) on branch $($wt.Branch) -> $($wt.Agent)" -ForegroundColor Green
    }
}

switch ($Action) {
    "init-wave1" {
        Initialize-WorktreeSet -Worktrees $Wave1Worktrees
    }
    "init-wave2" {
        Initialize-WorktreeSet -Worktrees $Wave2Worktrees
    }
    "init" {
        Initialize-WorktreeSet -Worktrees ($Wave1Worktrees + $Wave2Worktrees)
    }
    "list" {
        git worktree list
    }
    "merge-wave1" {
        Set-GitAuthorHygiene -TargetDir $RepoRoot
        git checkout main
        foreach ($wt in $Wave1Worktrees) {
            Write-Host "Merging $($wt.Branch) into main..." -ForegroundColor Cyan
            git merge --no-ff $wt.Branch -m "Merge branch '$($wt.Branch)' into main"
        }
    }
    "merge-wave2" {
        Set-GitAuthorHygiene -TargetDir $RepoRoot
        git checkout main
        foreach ($wt in $Wave2Worktrees) {
            Write-Host "Merging $($wt.Branch) into main..." -ForegroundColor Cyan
            git merge --no-ff $wt.Branch -m "Merge branch '$($wt.Branch)' into main"
        }
    }
    "clean" {
        foreach ($wt in ($Wave1Worktrees + $Wave2Worktrees)) {
            $wtPath = Join-Path (Join-Path $RepoRoot ".worktrees") $wt.Name
            if (Test-Path $wtPath) {
                git worktree remove --force $wtPath
                Write-Host "[REMOVED] $wtPath" -ForegroundColor Yellow
            }
        }
        git worktree prune
    }
}
