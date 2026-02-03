# Notes for Claude

## Deployment

The web interface is hosted on GitHub Pages at https://alonamaloh.github.io/torquemada/

GitHub Pages is configured to deploy from the `gh-pages` branch, NOT from GitHub Actions.

### Source Code Structure

- **C++ engine code**: On `master` branch in `search/`, `core/`, `nn/`, `tablebase/`, `web/src/wasm/`
- **JavaScript web UI**: On `gh-pages` branch (`*.js`, `index.html`, `style.css`)
- **Built WASM files**: On both branches (`web/dist/` on master, root on gh-pages)

### After changing WASM engine (C++ code on master)

```bash
make wasm                    # Build the WASM files
git add web/dist/engine.js web/dist/engine.wasm
git commit -m "..."
git push

# Deploy to gh-pages
git checkout gh-pages
git pull origin gh-pages
git checkout master -- web/dist/engine.js web/dist/engine.wasm
cp web/dist/engine.js . && cp web/dist/engine.wasm . && rm -rf web
git add engine.js engine.wasm
git commit -m "Update engine with <description>"
git push origin gh-pages
git checkout master
```

### After changing JavaScript UI (on gh-pages)

```bash
git checkout gh-pages
# Make changes to *.js, index.html, etc.
git add <files>
git commit -m "..."
git push origin gh-pages
git checkout master
```

The site typically updates within 1-2 minutes after pushing to `gh-pages`.
