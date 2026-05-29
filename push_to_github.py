import os
import subprocess
import time

REPO_URL = "https://github.com/JAHNAVISINDHU/ai-academic-adviser.git"

COMMITS = [
    ("chore: Initial commit with gitignore and env template", [".gitignore", ".env.example"]),
    ("docs: Add comprehensive project README", ["README.md"]),
    ("docs: Add system architecture diagram", ["docs/memory_architecture.png"]),
    ("chore(mcp): Add requirements for MCP server", ["mcp_server/requirements.txt"]),
    ("build(mcp): Create Dockerfile for FastAPI server", ["mcp_server/Dockerfile"]),
    ("feat(mcp): Initialize MCP app package", ["mcp_server/app/__init__.py"]),
    ("feat(mcp): Define Pydantic memory schemas for validation", ["mcp_server/app/memory_schemas.py"]),
    ("feat(mcp): Implement SQLite SQLAlchemy ORM and CRUD operations", ["mcp_server/app/database.py"]),
    ("feat(mcp): Integrate ChromaDB for semantic vector storage", ["mcp_server/app/vector_store.py"]),
    ("feat(mcp): Implement core MCP memory tools logic", ["mcp_server/app/tools.py"]),
    ("feat(mcp): Expose FastAPI endpoints for MCP server", ["mcp_server/app/main.py"]),
    ("chore(agent): Add dependencies for LLM agent", ["agent/requirements.txt"]),
    ("build(agent): Create agent Dockerfile", ["agent/Dockerfile"]),
    ("feat(agent): Implement interactive Claude CLI agent with MCP client", ["agent/agent.py"]),
    ("build: Orchestrate services using docker-compose", ["docker-compose.yml"]),
    ("test: Add evaluation submission data", ["submission.json"])
]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("Starting automated Git commit sequence...")
    
    if not os.path.exists(".git"):
        run_cmd("git init")
        run_cmd("git branch -M main")
    
    for message, files in COMMITS:
        for file in files:
            file_path = os.path.join(os.getcwd(), os.path.normpath(file))
            if os.path.exists(file_path):
                run_cmd(f'git add "{file_path}"')
        
        status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if status.stdout.strip():
            run_cmd(f'git commit -m "{message}"')
            time.sleep(0.5)
    
    run_cmd("git add .")
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if status.stdout.strip():
        run_cmd('git commit -m "chore: Final polish and remaining project files"')

    print(f"Linking to remote: {REPO_URL}")
    subprocess.run("git remote remove origin", shell=True, stderr=subprocess.DEVNULL)
    run_cmd(f"git remote add origin {REPO_URL}")
    print("Pushing to GitHub...")
    run_cmd("git push -u origin main --force")
    print("Successfully pushed to GitHub!")

if __name__ == "__main__":
    main()
