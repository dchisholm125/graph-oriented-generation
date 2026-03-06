import os
import subprocess
import shutil

class OpenCodeClient:
    """Connector for the OpenCode CLI using the @mention context system."""
    
    def __init__(self, binary="opencode"):
        self.binary = binary
        self.is_present = shutil.which(binary) is not None

    def complete(self, prompt, context_files=None):
        """Sends a prompt with @mention file context to the OpenCode CLI."""
        if not self.is_present:
            return f"[Error] '{self.binary}' CLI not found in PATH."

        # Build command: 'opencode run "the prompt" @file1 @file2 ...'
        cmd = [self.binary, "run", prompt]
        
        if context_files:
            for file in context_files:
                if os.path.exists(file):
                    cmd.append(f"@{file}")

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if process.returncode != 0:
                return f"[Error calling '{self.binary}'] {process.stderr.strip()}"
            
            return process.stdout.strip()
        except Exception as e:
            return f"[Error calling '{self.binary}'] {e}"

if __name__ == "__main__":
    client = OpenCodeClient()
    if client.is_present:
        print("Testing OpenCode CLI...")
        print(client.complete("Explain this project.", context_files=["README.md"]))
    else:
        print("OpenCode CLI not found.")
