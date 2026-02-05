# Notes for Claude

## Deployment

The web interface is hosted on GitHub Pages at https://alonamaloh.github.io/torquemada/

GitHub Pages is configured to deploy from the `gh-pages` branch, NOT from GitHub Actions.

### Directory Structure

- **Main repo** (`/home/alvaro/claude/torquemada`): Contains `master` branch with C++ source code
- **gh-pages worktree** (`/home/alvaro/claude/torquemada-gh-pages`): Contains `gh-pages` branch with deployed website

### Source Code Locations

- **C++ engine code**: `master` branch in `search/`, `core/`, `nn/`, `tablebase/`, `web/src/wasm/`
- **JavaScript web UI**: `gh-pages` branch (in worktree `../torquemada-gh-pages/`)
- **Built WASM files**: `web/dist/` on master after running `make wasm`

### After changing WASM engine (C++ code)

```bash
# 1. Build WASM (in main repo)
make wasm

# 2. Copy to gh-pages worktree and deploy
cp web/dist/engine.js web/dist/engine.wasm web/dist/engine.worker.js ../torquemada-gh-pages/
cd ../torquemada-gh-pages
git add engine.js engine.wasm engine.worker.js
git commit -m "Update engine: <description>"
git push origin gh-pages
cd ../torquemada

# 3. Optionally commit source changes to master
git add <changed-files>
git commit -m "..."
git push origin master
```

### After changing JavaScript UI

```bash
# 1. Edit files directly in the worktree
cd ../torquemada-gh-pages
# Make changes to *.js, index.html, style.css, etc.
git add <files>
git commit -m "..."
git push origin gh-pages
cd ../torquemada
```

### Version Tracking

The engine has a version string (`ENGINE_VERSION` in `web/src/wasm/bindings.cpp`) that gets logged to console on init. Update this when making changes to help debug caching issues:

```cpp
#define ENGINE_VERSION "YYYY-MM-DD-vN"
```

The site typically updates within 1-2 minutes after pushing to `gh-pages`.

### Worktree Management

If the worktree gets deleted or needs recreation:
```bash
git worktree add ../torquemada-gh-pages gh-pages
```

To list worktrees:
```bash
git worktree list
```
