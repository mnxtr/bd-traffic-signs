#!/usr/bin/env python3
"""
Directory Organization Script for bd-traffic-signs project
Reorganizes scattered files into a clean, logical directory structure.

Features:
- Dry-run mode to preview changes
- Backup creation before reorganization
- Rollback capability
- Git-aware safety checks
- Detailed logging

Usage:
    python organize_directory.py --dry-run      # Preview changes
    python organize_directory.py --create-backup # Create backup and organize
    python organize_directory.py --verbose      # Organize with detailed output
    python organize_directory.py --rollback     # Undo previous organization
"""

import os
import shutil
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class DirectoryOrganizer:
    """Organizes project directory structure with safety features."""
    
    def __init__(self, project_root: str, verbose: bool = False):
        self.project_root = Path(project_root)
        self.verbose = verbose
        self.operation_log_file = self.project_root / ".organization_log.json"
        self.backup_dir = self.project_root / "backup_before_organization"
        
    def get_file_mappings(self) -> Dict[str, str]:
        """Define source to destination mappings for all files to be moved."""
        return {
            # Documentation - Research Papers (main research outputs)
            "RESEARCH_PAPER.md": "docs/research/RESEARCH_PAPER.md",
            "RESEARCH_PAPER.html": "docs/research/RESEARCH_PAPER.html",
            "RESEARCH_PAPER.pdf": "docs/research/RESEARCH_PAPER.pdf",
            "PREPRINT.md": "docs/research/PREPRINT.md",
            "PREPRINT.html": "docs/research/PREPRINT.html",
            "PREPRINT.pdf": "docs/research/PREPRINT.pdf",
            "PREPRINT_A4.pdf": "docs/research/PREPRINT_A4.pdf",
            "CSE_499B_RESEARCH_PAPER.md": "docs/research/CSE_499B_RESEARCH_PAPER.md",
            "CSE_499B_FINAL_REPORT.pdf": "docs/research/CSE_499B_FINAL_REPORT.pdf",
            "CSE_499B_REPORT.docx": "docs/research/CSE_499B_REPORT.docx",
            "CSE_499B_COMPLETE_SUMMARY.txt": "docs/research/CSE_499B_COMPLETE_SUMMARY.txt",
            "Traffic Signs.pdf": "docs/research/Traffic_Signs.pdf",
            
            # Documentation - Guides
            "NEXT_STEPS_GUIDE.md": "docs/guides/NEXT_STEPS_GUIDE.md",
            "README_NEXT_STEPS.md": "docs/guides/README_NEXT_STEPS.md",
            
            # Documentation - Reports/Status
            "CODEBASE_ANALYSIS.md": "docs/reports/CODEBASE_ANALYSIS.md",
            "PAPER_GENERATION_COMPLETE.md": "docs/reports/PAPER_GENERATION_COMPLETE.md",
            "PAPER_READY.md": "docs/reports/PAPER_READY.md",
            
            # Archive - Old README
            "README_old.md": "docs/archive/README_old.md",
            
            # Scripts - Utils
            "quick_complete_paper.sh": "scripts/utils/quick_complete_paper.sh",
        }
    
    def get_files_to_delete(self) -> List[str]:
        """List of temporary files to delete."""
        return [
            ".~lock.CSE_499B_REPORT.docx#",  # LibreOffice lock file
        ]
    
    def check_git_status(self) -> Tuple[bool, str]:
        """Check if there are uncommitted changes in git."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            uncommitted = result.stdout.strip()
            if uncommitted:
                return False, f"Uncommitted changes detected:\n{uncommitted}"
            return True, "Git working directory is clean"
        except Exception as e:
            return True, f"Git check skipped: {e}"
    
    def create_backup(self) -> bool:
        """Create a backup of files before reorganization."""
        try:
            if self.backup_dir.exists():
                print(f"⚠️  Backup directory already exists: {self.backup_dir}")
                response = input("Overwrite existing backup? (y/n): ")
                if response.lower() != 'y':
                    return False
                shutil.rmtree(self.backup_dir)
            
            print(f"📦 Creating backup in {self.backup_dir}...")
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files that will be moved
            mappings = self.get_file_mappings()
            for source in mappings.keys():
                source_path = self.project_root / source
                if source_path.exists():
                    dest_path = self.backup_dir / source
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    if self.verbose:
                        print(f"  Backed up: {source}")
            
            print(f"✅ Backup created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def preview_changes(self) -> None:
        """Preview all changes that will be made."""
        print("\n" + "="*80)
        print("📋 PREVIEW: Changes that will be made")
        print("="*80 + "\n")
        
        mappings = self.get_file_mappings()
        files_to_delete = self.get_files_to_delete()
        
        # Group by destination directory
        by_category = {}
        for source, dest in mappings.items():
            category = str(Path(dest).parent)
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((source, dest))
        
        for category in sorted(by_category.keys()):
            print(f"\n📁 {category}/")
            for source, dest in by_category[category]:
                source_path = self.project_root / source
                status = "✓" if source_path.exists() else "✗ MISSING"
                print(f"   {status} {source}")
        
        if files_to_delete:
            print(f"\n🗑️  Files to delete:")
            for file in files_to_delete:
                file_path = self.project_root / file
                status = "✓" if file_path.exists() else "✗ MISSING"
                print(f"   {status} {file}")
        
        # Summary
        print("\n" + "-"*80)
        print(f"📊 Summary:")
        print(f"   Files to move: {len(mappings)}")
        print(f"   Files to delete: {len(files_to_delete)}")
        print(f"   New directories: {len(set(Path(d).parts[0] for d in mappings.values()))}")
        print("-"*80 + "\n")
    
    def organize(self, dry_run: bool = False) -> bool:
        """Execute the directory organization."""
        if dry_run:
            self.preview_changes()
            return True
        
        operations = []
        mappings = self.get_file_mappings()
        files_to_delete = self.get_files_to_delete()
        
        try:
            print("\n🚀 Starting directory organization...\n")
            
            # Create all destination directories
            all_dest_dirs = set(str(Path(dest).parent) for dest in mappings.values())
            for dest_dir in sorted(all_dest_dirs):
                dest_path = self.project_root / dest_dir
                dest_path.mkdir(parents=True, exist_ok=True)
                if self.verbose:
                    print(f"📁 Created directory: {dest_dir}")
            
            # Move files
            for source, dest in mappings.items():
                source_path = self.project_root / source
                dest_path = self.project_root / dest
                
                if not source_path.exists():
                    print(f"⚠️  Skipping {source} (not found)")
                    continue
                
                if dest_path.exists():
                    print(f"⚠️  Warning: {dest} already exists, skipping")
                    continue
                
                # Move the file
                shutil.move(str(source_path), str(dest_path))
                operations.append({
                    "type": "move",
                    "source": source,
                    "destination": dest,
                    "timestamp": datetime.now().isoformat()
                })
                
                status = "✓" if self.verbose else "→"
                print(f"  {status} {source} → {dest}")
            
            # Delete temporary files
            for file in files_to_delete:
                file_path = self.project_root / file
                if file_path.exists():
                    file_path.unlink()
                    operations.append({
                        "type": "delete",
                        "file": file,
                        "timestamp": datetime.now().isoformat()
                    })
                    print(f"  🗑️  Deleted: {file}")
            
            # Save operation log
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "operations": operations
            }
            with open(self.operation_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"\n✅ Organization complete! Moved {len([o for o in operations if o['type'] == 'move'])} files")
            print(f"📝 Operation log saved to: {self.operation_log_file}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during organization: {e}")
            print("💡 Use --rollback to undo partial changes")
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous state using operation log."""
        if not self.operation_log_file.exists():
            print("❌ No operation log found. Cannot rollback.")
            return False
        
        try:
            with open(self.operation_log_file, 'r') as f:
                log_data = json.load(f)
            
            operations = log_data.get("operations", [])
            print(f"\n🔄 Rolling back {len(operations)} operations...")
            
            # Reverse the operations
            for op in reversed(operations):
                if op["type"] == "move":
                    source_path = self.project_root / op["source"]
                    dest_path = self.project_root / op["destination"]
                    
                    if dest_path.exists():
                        # Move back
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(dest_path), str(source_path))
                        print(f"  ↩️  {op['destination']} → {op['source']}")
                    else:
                        print(f"  ⚠️  Skipping {op['destination']} (not found)")
                
                elif op["type"] == "delete":
                    print(f"  ⚠️  Cannot restore deleted file: {op['file']}")
            
            # Remove empty directories
            mappings = self.get_file_mappings()
            all_dest_dirs = set(str(Path(dest).parent) for dest in mappings.values())
            for dest_dir in sorted(all_dest_dirs, reverse=True):
                dest_path = self.project_root / dest_dir
                if dest_path.exists() and not any(dest_path.iterdir()):
                    dest_path.rmdir()
                    if self.verbose:
                        print(f"  🗑️  Removed empty directory: {dest_dir}")
            
            # Remove operation log
            self.operation_log_file.unlink()
            
            print("\n✅ Rollback complete!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error during rollback: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify that organization was successful."""
        print("\n🔍 Verifying organization...\n")
        
        mappings = self.get_file_mappings()
        missing = []
        found = []
        
        for source, dest in mappings.items():
            dest_path = self.project_root / dest
            if dest_path.exists():
                found.append(dest)
            else:
                source_path = self.project_root / source
                if source_path.exists():
                    missing.append(f"{source} (not moved yet)")
                else:
                    missing.append(f"{dest} (missing)")
        
        print(f"✓ Files in correct location: {len(found)}")
        if missing:
            print(f"✗ Files not found: {len(missing)}")
            for file in missing[:10]:
                print(f"  - {file}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        
        print()
        return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Organize bd-traffic-signs directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python organize_directory.py --dry-run         # Preview changes
  python organize_directory.py --create-backup   # Create backup and organize
  python organize_directory.py --verbose         # Organize with detailed output
  python organize_directory.py --rollback        # Undo previous organization
  python organize_directory.py --verify          # Verify organization
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing"
    )
    
    parser.add_argument(
        "--create-backup",
        action="store_true",
        help="Create backup before organizing"
    )
    
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous state"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify organization is complete"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        help="Skip git status check"
    )
    
    args = parser.parse_args()
    
    # Get project root (script location)
    project_root = Path(__file__).parent
    organizer = DirectoryOrganizer(project_root, verbose=args.verbose)
    
    # Handle rollback
    if args.rollback:
        organizer.rollback()
        return
    
    # Handle verify
    if args.verify:
        organizer.verify()
        return
    
    # Check git status
    if not args.skip_git_check and not args.dry_run:
        clean, message = organizer.check_git_status()
        if not clean:
            print(f"⚠️  {message}")
            print("\n💡 Commit your changes or use --skip-git-check to proceed anyway")
            response = input("\nProceed anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Create backup if requested
    if args.create_backup and not args.dry_run:
        if not organizer.create_backup():
            print("Backup creation failed. Aborting.")
            return
    
    # Execute organization
    success = organizer.organize(dry_run=args.dry_run)
    
    if success and not args.dry_run:
        print("\n📌 Next steps:")
        print("  1. Review the changes: git status")
        print("  2. Verify organization: python organize_directory.py --verify")
        print("  3. If satisfied, commit: git add . && git commit -m 'Organize directory structure'")
        print("  4. If issues, rollback: python organize_directory.py --rollback")


if __name__ == "__main__":
    main()
