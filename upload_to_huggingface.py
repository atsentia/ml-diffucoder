#!/usr/bin/env python3
"""Upload JAX DiffuCoder model to HuggingFace Hub."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm

# Load .env file if it exists
from dotenv import load_dotenv
env_path = Path(__file__).parent / "jax_lm" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")

console = Console()

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def upload_model(
    local_dir: Path,
    repo_id: str,
    token: str = None,
    create_pr: bool = False,
    skip_confirm: bool = False
):
    """Upload model directory to HuggingFace Hub."""
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        console.print(f"[green]✓[/green] Repository ready: {repo_id}")
    except Exception as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
    
    # List files to upload
    files_to_upload = []
    total_size = 0
    
    for file_path in local_dir.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            size = file_path.stat().st_size
            total_size += size
            files_to_upload.append(file_path)
    
    console.print(f"\n[cyan]Files to upload:[/cyan] {len(files_to_upload)}")
    console.print(f"[cyan]Total size:[/cyan] {total_size / 1e9:.2f} GB")
    
    # Confirm upload
    if not skip_confirm:
        if not Confirm.ask("\nProceed with upload?"):
            console.print("[yellow]Upload cancelled[/yellow]")
            return
    
    # Upload files with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        upload_task = progress.add_task(
            "Uploading files...", 
            total=len(files_to_upload)
        )
        
        # Upload in batches
        batch_size = 10
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i:i + batch_size]
            
            # Upload batch
            try:
                # Upload each file individually
                for file_path in batch:
                    path_in_repo = str(file_path.relative_to(local_dir))
                    
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        token=token,
                        create_pr=create_pr,
                        commit_message=f"Upload {path_in_repo}"
                    )
                
                progress.update(upload_task, advance=len(batch))
                
            except Exception as e:
                console.print(f"\n[red]Error uploading batch: {e}[/red]")
                return
    
    console.print(f"\n[bold green]✅ Upload complete![/bold green]")
    console.print(f"[green]View your model at:[/green] https://huggingface.co/{repo_id}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload JAX DiffuCoder model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/DiffuCoder-7B-JAX"),
        help="Path to model directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="atsentia/DiffuCoder-7B-JAX",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create pull request instead of direct push"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]Error: No HuggingFace token provided[/red]")
        console.print("Set HF_TOKEN environment variable or use --token")
        return 1
    
    # Check model directory
    if not args.model_dir.exists():
        console.print(f"[red]Error: Model directory not found: {args.model_dir}[/red]")
        return 1
    
    console.rule("[bold cyan]HuggingFace Model Uploader[/bold cyan]")
    console.print(f"\n[bold]Model:[/bold] {args.model_dir}")
    console.print(f"[bold]Repository:[/bold] {args.repo_id}")
    
    # Upload
    upload_model(
        args.model_dir,
        args.repo_id,
        token=token,
        create_pr=args.create_pr,
        skip_confirm=args.yes
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())