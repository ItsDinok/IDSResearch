#include <windows.h>
#include <psapi.h>
#include <stdio.h>

// This identifies the process
DWORD GetScriptPID(const char* script_name) {
    char cmd[512];
    // Powershell command that finds python.exe processes
	snprintf(cmd, sizeof(cmd),
        "powershell -Command \"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*%s*' -and $_.Name -eq 'python.exe' } | Select-Object -ExpandProperty ProcessId\"", script_name);

	// Opens a pipe to read output of powershell command
    FILE* pipe = _popen(cmd, "r");
    if (!pipe) {
        printf("Failed to run PowerShell.\n");
        return 0;
    }

	// Holds output of pipe and pid when found
    char buffer[128];
    DWORD pid = 0;

	// Reds each line of output and parses the pid from output
    while (fgets(buffer, sizeof(buffer), pipe)) {
        if (sscanf(buffer, "%lu", &pid) == 1) {
            break; // Grab the first match
        }
    }

	// Closes pipe and returns
    _pclose(pipe);
    return pid;
}

int main() {
	// Get PID of process
    const char* script_name = "node1.py";
    DWORD pid = GetScriptPID(script_name);

    if (pid == 0) {
        printf("Could not find any python.exe running '%s'\n", script_name);
        return 1;
    }

	// Open process
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (hProcess == NULL) {
        printf("Failed to open process %lu\n", pid);
        return 1;
    }

    PROCESS_MEMORY_COUNTERS pmc;
    SIZE_T maxMemory = 0;

    printf("Monitoring memory usage of PID %lu (script: %s)...\nPress Ctrl+C to stop.\n", pid, script_name);

	// Calculate memory every 0.5 seconds
    while (1) {
        if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
            SIZE_T currentUsage = pmc.WorkingSetSize;
            if (currentUsage > maxMemory) {
                maxMemory = currentUsage;
            }

            printf("Current: %.2f MB | Max: %.2f MB\r",
                currentUsage / (1024.0 * 1024.0),
                maxMemory / (1024.0 * 1024.0));
            fflush(stdout);
        } else {
            printf("\nFailed to get memory info. Exiting.\n");
            break;
        }
        Sleep(5000);
    }

    CloseHandle(hProcess);
    printf("\nMax memory usage: %.2f MB\n", maxMemory / (1024.0 * 1024.0));
    return 0;
}

