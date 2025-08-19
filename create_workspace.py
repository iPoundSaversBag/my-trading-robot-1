#!/usr/bin/env python3
"""
Force VS Code to detect GitHub repository
"""

import os
import json

def create_vscode_workspace():
    """Create VS Code workspace file to help with GitHub detection"""
    
    workspace_config = {
        "folders": [
            {
                "name": "Trading Robot",
                "path": "."
            }
        ],
        "settings": {
            "git.enabled": True,
            "git.autoRepositoryDetection": "openEditors",
            "github.gitAuthentication": True,
            "github-actions.workflows.pinned.workflows": [
                "bidirectional-sync.yml",
                "trading-bot.yml", 
                "deploy.yml"
            ]
        },
        "extensions": {
            "recommendations": [
                "github.vscode-github-actions",
                "github.vscode-pull-request-github"
            ]
        }
    }
    
    with open("trading-robot.code-workspace", "w") as f:
        json.dump(workspace_config, f, indent=2)
    
    print("✅ Created VS Code workspace file")
    print("💡 Open this file in VS Code to properly detect GitHub repository")

if __name__ == "__main__":
    create_vscode_workspace()
