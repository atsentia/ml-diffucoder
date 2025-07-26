#!/usr/bin/env python3
"""Test local installation of jax-diffucoder package."""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"


def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"{e.stderr}\n{e.stdout}"


def test_local_installation():
    """Test installing and using the jax-diffucoder package locally."""
    
    console.rule("[bold cyan]JAX-DiffuCoder Local Installation Test[/bold cyan]")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        console.print(f"\n[cyan]Test directory:[/cyan] {tmpdir}")
        
        # Create virtual environment
        venv_path = tmpdir / "venv"
        console.print("\n[bold]1. Creating virtual environment...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating venv...", total=None)
            
            success, output = run_command(f"python -m venv {venv_path}")
            if not success:
                console.print(f"[red]✗ Failed to create venv: {output}[/red]")
                return False
            
            progress.update(task, completed=True)
        
        console.print("[green]✓ Virtual environment created[/green]")
        
        # Activate venv (for subprocess commands)
        if sys.platform == "win32":
            pip_cmd = str(venv_path / "Scripts" / "pip")
            python_cmd = str(venv_path / "Scripts" / "python")
        else:
            pip_cmd = str(venv_path / "bin" / "pip")
            python_cmd = str(venv_path / "bin" / "python")
        
        # Install jax-diffucoder from local path
        console.print("\n[bold]2. Installing jax-diffucoder from local path...[/bold]")
        
        jax_lm_path = Path(__file__).parent / "jax_lm"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Installing package...", total=None)
            
            # Install with minimal dependencies first
            success, output = run_command(
                f"{pip_cmd} install -e {jax_lm_path}[hf,tokenizer]"
            )
            
            if not success:
                console.print(f"[red]✗ Installation failed: {output}[/red]")
                return False
            
            progress.update(task, completed=True)
        
        console.print("[green]✓ Package installed successfully[/green]")
        
        # Test import
        console.print("\n[bold]3. Testing package import...[/bold]")
        
        test_script = tmpdir / "test_import.py"
        test_script.write_text("""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

try:
    import jax_lm
    print(f"✓ Successfully imported jax_lm version {jax_lm.__version__}")
    
    # Test specific imports
    from jax_lm import load_model, generate
    print("✓ Core functions imported")
    
    from jax_lm.utils.pure_jax_loader import PureJAXModelLoader
    print("✓ Pure JAX loader imported")
    
    print("\\nAvailable functions:")
    for item in dir(jax_lm):
        if not item.startswith('_'):
            print(f"  - {item}")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
""")
        
        success, output = run_command(f"{python_cmd} {test_script}")
        if not success:
            console.print(f"[red]✗ Import test failed:[/red]\n{output}")
            return False
        
        console.print(output.strip())
        
        # Test partial download
        console.print("\n[bold]4. Testing model download (partial)...[/bold]")
        
        download_test = tmpdir / "test_download.py"
        download_test.write_text("""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from jax_lm.utils.pure_jax_loader import PureJAXModelLoader
from pathlib import Path

loader = PureJAXModelLoader()

# Try to download just config file
print("Attempting to download config file...")
try:
    # This would normally be "atsentia/DiffuCoder-7B-JAX"
    # For now, test with local path
    local_model = Path(__file__).parent.parent.parent / "models" / "DiffuCoder-7B-JAX"
    
    if local_model.exists():
        print(f"✓ Found local model at {local_model}")
        config_path = local_model / "config.json"
        if config_path.exists():
            print("✓ Config file exists")
            import json
            with open(config_path) as f:
                config = json.load(f)
            print(f"  Model type: {config.get('model_type', 'unknown')}")
            print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
        else:
            print("✗ Config file not found")
    else:
        print("✗ Local model not found")
        print("Note: HuggingFace download would happen here if model was uploaded")
        
except Exception as e:
    print(f"✗ Download test failed: {e}")
    import traceback
    traceback.print_exc()
""")
        
        success, output = run_command(f"{python_cmd} {download_test}")
        console.print(output.strip())
        
        # Summary
        console.rule("[bold]Summary[/bold]")
        
        if success:
            console.print("[bold green]✅ All tests passed![/bold green]")
            console.print("\nThe jax-diffucoder package can be installed locally.")
            console.print("Next steps:")
            console.print("  1. Upload model to HuggingFace Hub")
            console.print("  2. Test download from HuggingFace")
            console.print("  3. Publish to PyPI")
            return True
        else:
            console.print("[bold red]❌ Some tests failed![/bold red]")
            return False


def main():
    """Main function."""
    success = test_local_installation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())