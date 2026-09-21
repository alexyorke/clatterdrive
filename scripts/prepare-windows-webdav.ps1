$ErrorActionPreference = "Stop"

# Only provision disposable CI/explicit installer-test VMs, never silently
# enable Windows features or services on a developer's ordinary workstation.
if ($env:GITHUB_ACTIONS -ne "true" -and $env:CLATTERDRIVE_INSTALLER_E2E_VM -ne "1") {
    throw "WebDAV provisioning is restricted to CI or an explicit installer-test VM."
}

$webClient = Get-Service -Name WebClient -ErrorAction SilentlyContinue
if ($null -eq $webClient) {
    if (-not (Get-Command Install-WindowsFeature -ErrorAction SilentlyContinue)) {
        throw "WebClient is absent and Windows Server feature installation is unavailable."
    }
    $result = Install-WindowsFeature -Name WebDAV-Redirector
    if (-not $result.Success) {
        throw "WebDAV-Redirector installation failed: $($result.ExitCode)"
    }
    if ($result.RestartNeeded -eq "Yes") {
        # Some runner images report a pending reboot even when the redirector
        # can start now. Require a running service AND the actual mapped-drive
        # E2E below; never turn this into a skipped/passing capability check.
        Write-Warning "WebDAV-Redirector requested a reboot; testing whether WebClient can start in this session."
    }
}

Set-Service -Name WebClient -StartupType Manual
Start-Service -Name WebClient
$webClient = Get-Service -Name WebClient
$webClient.WaitForStatus("Running", [TimeSpan]::FromSeconds(30))
Write-Output "WebClient is running; mapped-drive E2E remains enabled."
