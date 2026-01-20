# RAG Application

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-FF4B4B.svg)](https://streamlit.io/)

Retrieval-Augmented Generation application for document-based question answering. The system enables uploading documents, processing them into vector embeddings, and querying using natural language with LLM-powered responses.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Technology Stack](#technology-stack)
- [Quick Start Guide](#quick-start-guide)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The application consists of three main components:

1. **FastAPI Backend** - REST API server handling document ingestion, vector search, and chat streaming
2. **Streamlit Frontend** - Web interface for document upload and interactive Q&A
3. **Inngest Workflows** - Background job processing for async document ingestion

Data flow:
```
User Upload → Inngest Workflow → Document Parsing → Embedding Generation → Qdrant Storage
User Query → Embedding → Vector Search → Context Retrieval → LLM Response → Streaming Output
```

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Runtime | Python 3.13+ | Application runtime |
| Backend | FastAPI, Uvicorn | REST API server |
| Frontend | Streamlit | Web interface |
| Embeddings | Sentence Transformers | Text vectorization (all-MiniLM-L6-v2) |
| LLM | Groq API | Chat completions (Llama, Mixtral, DeepSeek) |
| Vector DB | Qdrant | Semantic search storage |
| Workflows | Inngest | Background job processing |
| Package Manager | uv | Fast dependency management |

---

## Quick Start Guide

Choose your operating system and follow the step-by-step instructions.

---

### Linux

#### Step 1: Install Python 3.13

**Ubuntu/Debian:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev
```

**Fedora:**
```bash
sudo dnf install python3.13
```

**Arch Linux:**
```bash
sudo pacman -S python
```

Verify installation:
```bash
python3.13 --version
```

#### Step 2: Install uv (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal or run:
```bash
source ~/.bashrc   # or ~/.zshrc if using zsh
```

Verify:
```bash
uv --version
```

#### Step 3: Install Docker

**Ubuntu/Debian:**
```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt update
sudo apt install ca-certificates curl gnupg

# Add Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Run Docker without sudo (optional, requires logout/login)
sudo usermod -aG docker $USER
```

**Fedora:**
```bash
sudo dnf install docker docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
```

Verify:
```bash
docker --version
```

#### Step 4: Install Node.js (for Inngest CLI)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
```

Verify:
```bash
node --version
npm --version
```

#### Step 5: Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create an account or log in
3. Navigate to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the generated key (starts with `gsk_`)

#### Step 6: Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/SebastianCielma/RAG.git
cd RAG

# Install dependencies
uv sync

# Create environment file
cat > .env << EOF
GROQ_API_KEY=gsk_your_api_key_here
EOF
```

**Replace `gsk_your_api_key_here` with your actual Groq API key.**

#### Step 7: Run the Application

Open 4 terminal windows/tabs in the RAG directory:

**Terminal 1 - Start Qdrant:**
```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Terminal 2 - Start Inngest:**
```bash
npx inngest-cli@latest dev
```

**Terminal 3 - Start Backend:**
```bash
uv run uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 4 - Start Frontend:**
```bash
uv run streamlit run frontend/app.py --server.port 8501
```

Open your browser at: **http://localhost:8501**

---

### macOS

#### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the instructions to add Homebrew to your PATH.

#### Step 2: Install Python 3.13

```bash
brew install python@3.13
```

Verify:
```bash
python3.13 --version
```

#### Step 3: Install uv (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal or run:
```bash
source ~/.zshrc   # or ~/.bashrc
```

Verify:
```bash
uv --version
```

#### Step 4: Install Docker Desktop

1. Download Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Open the downloaded `.dmg` file
3. Drag Docker to Applications folder
4. Open Docker from Applications
5. Wait for Docker to start (whale icon in menu bar becomes stable)

Verify:
```bash
docker --version
```

#### Step 5: Install Node.js (for Inngest CLI)

```bash
brew install node
```

Verify:
```bash
node --version
npm --version
```

#### Step 6: Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create an account or log in
3. Navigate to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the generated key (starts with `gsk_`)

#### Step 7: Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/SebastianCielma/RAG.git
cd RAG

# Install dependencies
uv sync

# Create environment file
cat > .env << EOF
GROQ_API_KEY=gsk_your_api_key_here
EOF
```

**Replace `gsk_your_api_key_here` with your actual Groq API key.**

#### Step 8: Run the Application

Open 4 terminal windows/tabs (Cmd+T) in the RAG directory:

**Terminal 1 - Start Qdrant:**
```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Terminal 2 - Start Inngest:**
```bash
npx inngest-cli@latest dev
```

**Terminal 3 - Start Backend:**
```bash
uv run uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 4 - Start Frontend:**
```bash
uv run streamlit run frontend/app.py --server.port 8501
```

Open your browser at: **http://localhost:8501**

---

### Windows

#### Step 1: Install Python 3.13

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.13.x installer
3. Run the installer
4. **Check "Add Python to PATH"** at the bottom of the installer window
5. Click "Install Now"

Verify in PowerShell or Command Prompt:
```powershell
python --version
```

#### Step 2: Install uv (Package Manager)

Open PowerShell as Administrator and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell.

Verify:
```powershell
uv --version
```

#### Step 3: Install Docker Desktop

1. Download Docker Desktop from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Run the installer
3. **Enable WSL 2 backend** when prompted (recommended)
4. Restart your computer when prompted
5. After restart, open Docker Desktop and wait for it to start

Verify in PowerShell:
```powershell
docker --version
```

**Note:** If Docker asks to enable WSL 2:
1. Open PowerShell as Administrator
2. Run: `wsl --install`
3. Restart computer
4. Open Docker Desktop again

#### Step 4: Install Node.js (for Inngest CLI)

1. Go to [nodejs.org](https://nodejs.org/)
2. Download LTS version installer
3. Run the installer with default options

Verify in PowerShell:
```powershell
node --version
npm --version
```

#### Step 5: Install Git (if not installed)

1. Go to [git-scm.com/download/win](https://git-scm.com/download/win)
2. Download and run installer
3. Use default options

#### Step 6: Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create an account or log in
3. Navigate to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the generated key (starts with `gsk_`)

#### Step 7: Clone and Setup Project

Open PowerShell and run:
```powershell
# Clone repository
git clone https://github.com/SebastianCielma/RAG.git
cd RAG

# Install dependencies
uv sync

# Create environment file
@"
GROQ_API_KEY=gsk_your_api_key_here
"@ | Out-File -FilePath .env -Encoding utf8
```

**Open the `.env` file and replace `gsk_your_api_key_here` with your actual Groq API key.**

#### Step 8: Run the Application

Open 4 PowerShell windows, navigate to RAG directory in each (`cd path\to\RAG`):

**PowerShell 1 - Start Qdrant:**
```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**PowerShell 2 - Start Inngest:**
```powershell
npx inngest-cli@latest dev
```

**PowerShell 3 - Start Backend:**
```powershell
uv run uvicorn rag.main:app --reload --host 0.0.0.0 --port 8000
```

**PowerShell 4 - Start Frontend:**
```powershell
uv run streamlit run frontend/app.py --server.port 8501
```

Open your browser at: **http://localhost:8501**

---

## Configuration

All configuration is done through environment variables in the `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | API key from console.groq.com |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | No | `docs` | Collection name |
| `EMBED_MODEL` | No | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `CHUNK_SIZE` | No | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | No | `200` | Overlap between chunks |
| `LLM_TEMPERATURE` | No | `0.2` | LLM response randomness |

---

## Running the Application

After completing the Quick Start for your OS, you need to run 4 services:

| Service | Port | Command | Purpose |
|---------|------|---------|---------|
| Qdrant | 6333 | `docker run ...` | Vector database |
| Inngest | 8288 | `npx inngest-cli@latest dev` | Workflow orchestration |
| Backend | 8000 | `uv run uvicorn rag.main:app ...` | REST API |
| Frontend | 8501 | `uv run streamlit run frontend/app.py ...` | Web interface |

**Start order matters:** Qdrant → Inngest → Backend → Frontend

### Stopping the Application

1. Press `Ctrl+C` in each terminal to stop services
2. Stop Qdrant container: `docker stop qdrant`
3. Remove Qdrant container (optional): `docker rm qdrant`

### Restarting After First Setup

```bash
# Linux/macOS
docker start qdrant                                              # Terminal 1
npx inngest-cli@latest dev                                       # Terminal 2
uv run uvicorn rag.main:app --reload --port 8000                # Terminal 3
uv run streamlit run frontend/app.py --server.port 8501          # Terminal 4
```

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/chat` | Streaming chat with RAG |

Interactive API docs available at: **http://localhost:8000/docs**

### Available LLM Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| Llama 3.3 70B | Medium | Highest | Complex analysis |
| Llama 3.1 8B | Fast | Good | Quick queries |
| Mixtral 8x7B | Medium | High | Balanced use |
| DeepSeek R1 70B | Slow | Highest | Reasoning tasks |
| Qwen QWQ 32B | Medium | High | Reasoning |

---

## Project Structure

```
RAG/
├── src/rag/                 # Backend application
│   ├── main.py              # FastAPI entry point
│   ├── core/                # Config, exceptions
│   ├── models/              # Pydantic schemas
│   ├── services/            # Business logic
│   ├── db/                  # Qdrant client
│   └── workflows/           # Inngest functions
├── frontend/                # Streamlit UI
│   └── app.py
├── tests/                   # Test suite
├── pyproject.toml           # Project config
└── .env                     # Environment variables
```

---

## Development

### Install Dev Dependencies

```bash
uv sync --group dev
```

### Run Tests

```bash
uv run pytest
```

### Linting and Formatting

```bash
uv run ruff check .          # Check for issues
uv run ruff check . --fix    # Auto-fix issues
uv run ruff format .         # Format code
```

### Type Checking

```bash
uv run mypy src/
```

---

## Troubleshooting

### "Docker daemon not running"

- **Linux:** `sudo systemctl start docker`
- **macOS/Windows:** Open Docker Desktop application

### "Connection refused" on port 6333

Qdrant is not running. Start it with:
```bash
docker start qdrant
# or if container doesn't exist:
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### "ModuleNotFoundError" when running

Dependencies not installed. Run:
```bash
uv sync
```

### Inngest functions not appearing

1. Make sure Inngest CLI is running (`npx inngest-cli@latest dev`)
2. Make sure Backend is running
3. Check http://127.0.0.1:8288 - functions should appear under "Apps"

### First run is slow

The embedding model (~100MB) downloads on first run. This is normal and cached for subsequent runs.

### "Invalid API Key" from Groq

1. Check `.env` file has correct key
2. Key should start with `gsk_`
3. No quotes around the key
4. Verify at [console.groq.com](https://console.groq.com)

### Port already in use

Change port in the startup command:
- Backend: `--port 8001` instead of `--port 8000`
- Frontend: `--server.port 8502` instead of `--server.port 8501`

---

**License:** MIT