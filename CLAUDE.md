# Notes for Claude

## Deployment

The web interface is hosted on GitHub Pages at https://alonamaloh.github.io/torquemada/

GitHub Pages is configured to deploy from the `gh-pages` branch, NOT from GitHub Actions.

After making changes to the WASM engine (`web/dist/engine.js`, `web/dist/engine.wasm`), you must manually update the `gh-pages` branch:

```bash
git checkout gh-pages
git pull origin gh-pages
git checkout master -- web/dist/engine.js web/dist/engine.wasm
git commit -m "Update engine with <description of changes>"
git push origin gh-pages
git checkout master
```

The site typically updates within 1-2 minutes after pushing to `gh-pages`.
