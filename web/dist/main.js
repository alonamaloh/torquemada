/**
 * Main entry point for the checkers web app
 */

import { GameController } from './game-controller.js';
import { getEngine } from './engine-api.js';
import { TablebaseLoader, loadNNModelFile } from './tablebase-loader.js';

// Global state
let gameController = null;
let tablebaseLoader = null;

/**
 * Initialize the application
 */
async function init() {
    // Get DOM elements
    const canvas = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const loadingEl = document.getElementById('loading');
    const gameContainerEl = document.getElementById('game-container');

    // Show loading state
    if (loadingEl) loadingEl.style.display = 'flex';
    if (gameContainerEl) gameContainerEl.style.display = 'none';

    try {
        // Initialize engine
        updateLoadingStatus('Initializing engine...');
        const engine = getEngine();
        await engine.init('./engine-worker.js');

        // Initialize tablebase loader
        updateLoadingStatus('Checking tablebases...');
        try {
            tablebaseLoader = new TablebaseLoader();
            await tablebaseLoader.init();

            // Check if tablebases are already stored
            const stored = await tablebaseLoader.checkStoredTablebases();
            if (stored.length > 0) {
                updateLoadingStatus(`Loading ${stored.length} tablebases...`);
                await loadTablebases();
            }
        } catch (err) {
            console.warn('OPFS not available:', err);
        }

        // Try to load NN models
        updateLoadingStatus('Loading neural network...');
        try {
            // Try loading from local files first
            const nnData = await loadNNModelFile('./models/model_005_long.bin');
            await engine.loadNNModel(nnData, false);
        } catch (err) {
            console.warn('Could not load NN model:', err);
        }

        try {
            const dtmNNData = await loadNNModelFile('./models/endgame_wdl.bin');
            await engine.loadNNModel(dtmNNData, true);
        } catch (err) {
            console.warn('Could not load DTM NN model:', err);
        }

        // Set up game controller
        updateLoadingStatus('Starting game...');
        gameController = new GameController(canvas, statusEl);
        await gameController.init();

        // Set up callbacks
        gameController.onMove = (move, board) => {
            updateMoveHistory();
        };

        gameController.onGameOver = (winner, reason) => {
            showGameOver(winner);
        };

        gameController.onThinkingStart = () => {
            setThinkingIndicator(true);
        };

        gameController.onThinkingEnd = () => {
            setThinkingIndicator(false);
        };

        gameController.onSearchInfo = (info) => {
            updateSearchInfo(info);
        };

        gameController.onModeChange = () => {
            updateModeButtons();
        };

        // Set up UI event handlers
        setupEventHandlers();
        updateModeButtons();

        // Resize board to fit
        resizeBoard();
        window.addEventListener('resize', resizeBoard);

        // Hide loading, show game
        if (loadingEl) loadingEl.style.display = 'none';
        if (gameContainerEl) gameContainerEl.style.display = 'flex';

        // Check engine status
        const status = await engine.getStatus();
        console.log('Engine status:', status);

    } catch (err) {
        console.error('Initialization failed:', err);
        updateLoadingStatus(`Error: ${err.message}`);
    }
}

/**
 * Update loading status message
 */
function updateLoadingStatus(message) {
    const el = document.getElementById('loading-status');
    if (el) el.textContent = message;
}

/**
 * Update toggle buttons to reflect current mode
 */
function updateModeButtons() {
    const btnEngineWhite = document.getElementById('btn-engine-white');
    const btnEngineBlack = document.getElementById('btn-engine-black');
    const btnTwoPlayer = document.getElementById('btn-two-player');

    if (!btnEngineWhite || !btnEngineBlack || !btnTwoPlayer) return;

    // Remove active from all
    btnEngineWhite.classList.remove('active');
    btnEngineBlack.classList.remove('active');
    btnTwoPlayer.classList.remove('active');

    // Set active based on humanColor
    // humanColor='black' means engine plays white
    // humanColor='white' means engine plays black
    // humanColor='both' means two-player mode
    switch (gameController.humanColor) {
        case 'black':
            btnEngineWhite.classList.add('active');
            break;
        case 'white':
            btnEngineBlack.classList.add('active');
            break;
        case 'both':
            btnTwoPlayer.classList.add('active');
            break;
    }
}

/**
 * Set up UI event handlers
 */
function setupEventHandlers() {
    // New game button
    const newGameBtn = document.getElementById('btn-new-game');
    if (newGameBtn) {
        newGameBtn.addEventListener('click', async () => {
            await gameController.newGame();
            updateModeButtons();
        });
    }

    // Undo button
    const undoBtn = document.getElementById('btn-undo');
    if (undoBtn) {
        undoBtn.addEventListener('click', async () => {
            await gameController.undo();
            updateMoveHistory();
        });
    }

    // Redo button
    const redoBtn = document.getElementById('btn-redo');
    if (redoBtn) {
        redoBtn.addEventListener('click', async () => {
            await gameController.redo();
            updateMoveHistory();
        });
    }

    // Flip board button
    const flipBtn = document.getElementById('btn-flip');
    if (flipBtn) {
        flipBtn.addEventListener('click', () => gameController.flipBoard());
    }

    // Mode toggle buttons
    const btnEngineWhite = document.getElementById('btn-engine-white');
    const btnEngineBlack = document.getElementById('btn-engine-black');
    const btnTwoPlayer = document.getElementById('btn-two-player');

    if (btnEngineWhite) {
        btnEngineWhite.addEventListener('click', () => {
            gameController.setHumanColor('black'); // human plays black = engine plays white
            updateModeButtons();
        });
    }
    if (btnEngineBlack) {
        btnEngineBlack.addEventListener('click', () => {
            gameController.setHumanColor('white'); // human plays white = engine plays black
            updateModeButtons();
        });
    }
    if (btnTwoPlayer) {
        btnTwoPlayer.addEventListener('click', () => {
            gameController.setHumanColor('both');
            updateModeButtons();
        });
    }

    // Strength buttons (Fast/Slow)
    const fastBtn = document.getElementById('btn-nodes-fast');
    const slowBtn = document.getElementById('btn-nodes-slow');
    if (fastBtn && slowBtn) {
        fastBtn.addEventListener('click', () => {
            gameController.setEngineParams(100, 100000);
            fastBtn.classList.add('active');
            slowBtn.classList.remove('active');
        });
        slowBtn.addEventListener('click', () => {
            gameController.setEngineParams(100, 1000000);
            slowBtn.classList.add('active');
            fastBtn.classList.remove('active');
        });
    }

    // Download tablebases button
    const downloadBtn = document.getElementById('btn-download-tb');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', showDownloadDialog);
    }
}

/**
 * Load tablebases from OPFS into engine
 */
async function loadTablebases() {
    if (!tablebaseLoader) return;

    const engine = getEngine();
    const tablebases = await tablebaseLoader.loadAllTablebases();

    for (const [materialKey, data] of tablebases) {
        await engine.loadTablebase(materialKey, data);
    }

    console.log(`Loaded ${tablebases.size} tablebases`);
}

/**
 * Show tablebase download dialog
 */
async function showDownloadDialog() {
    const dialog = document.getElementById('download-dialog');
    if (!dialog) {
        alert('Tablebase download not available');
        return;
    }

    if (!tablebaseLoader || !tablebaseLoader.isAvailable()) {
        alert('Tablebase storage requires OPFS (Origin Private File System), which requires HTTPS or localhost. Make sure you are using the HTTPS server.');
        return;
    }

    dialog.style.display = 'flex';

    const progressEl = dialog.querySelector('.progress');
    const statusEl = dialog.querySelector('.status');
    const cancelBtn = dialog.querySelector('.cancel-btn');

    let cancelled = false;
    cancelBtn.onclick = () => {
        cancelled = true;
        dialog.style.display = 'none';
    };

    try {
        await tablebaseLoader.downloadTablebases((loaded, total, file) => {
            if (cancelled) return;
            const pct = Math.round((loaded / total) * 100);
            if (progressEl) progressEl.style.width = `${pct}%`;
            if (statusEl) statusEl.textContent = `Downloading: ${file} (${loaded}/${total})`;
        });

        if (!cancelled) {
            statusEl.textContent = 'Loading into engine...';
            await loadTablebases();
            statusEl.textContent = 'Complete!';
            setTimeout(() => {
                dialog.style.display = 'none';
            }, 1000);
        }
    } catch (err) {
        if (statusEl) statusEl.textContent = `Error: ${err.message}`;
    }
}

/**
 * Update move history display
 * Shows played moves in white, undone moves (redo stack) in gray
 */
function updateMoveHistory() {
    const historyEl = document.getElementById('move-history');
    if (!historyEl || !gameController) return;

    const { history, redo } = gameController.getMoveHistoryForDisplay();
    const allMoves = [...history, ...redo];

    if (allMoves.length === 0) {
        historyEl.innerHTML = '';
        return;
    }

    let html = '';
    for (let i = 0; i < allMoves.length; i++) {
        const moveNum = Math.floor(i / 2) + 1;
        const isRedo = i >= history.length;

        if (i % 2 === 0) {
            html += `${moveNum}. `;
        }

        if (isRedo) {
            html += `<span class="redo-move">${allMoves[i]}</span> `;
        } else {
            html += `${allMoves[i]} `;
        }
    }

    historyEl.innerHTML = html.trim();
    historyEl.scrollTop = historyEl.scrollHeight;
}

/**
 * Show game over message
 */
function showGameOver(winner) {
    const message = winner === 'draw' ? 'Game drawn!' : `${winner.charAt(0).toUpperCase() + winner.slice(1)} wins!`;
    // Could show a modal or alert
    console.log(message);
}

/**
 * Set thinking indicator
 */
function setThinkingIndicator(thinking) {
    const indicator = document.getElementById('thinking-indicator');
    if (indicator) {
        indicator.style.display = thinking ? 'inline' : 'none';
    }

    // Show/hide search info panel
    const searchInfo = document.getElementById('search-info');
    if (searchInfo) {
        searchInfo.style.display = thinking ? 'block' : 'block';  // Keep visible after search
    }
}

/**
 * Update search info display
 */
function updateSearchInfo(info) {
    const searchInfo = document.getElementById('search-info');
    if (!searchInfo) return;

    searchInfo.style.display = 'block';

    const depthEl = document.getElementById('search-depth');
    const scoreEl = document.getElementById('search-score');
    const nodesEl = document.getElementById('search-nodes');
    const pvEl = document.getElementById('search-pv');

    if (depthEl) depthEl.textContent = info.depth || '-';
    if (scoreEl) scoreEl.textContent = info.scoreStr || '-';
    if (nodesEl) {
        // Format nodes with commas
        const nodes = info.nodes || 0;
        nodesEl.textContent = nodes.toLocaleString();
    }
    if (pvEl) pvEl.textContent = info.pvStr || '-';
}

/**
 * Resize board to fit container
 */
function resizeBoard() {
    const container = document.getElementById('board-container');
    const canvas = document.getElementById('board');

    if (container && canvas && gameController) {
        // Use width as primary dimension, with fallback to canvas default
        const containerWidth = container.clientWidth || 480;
        const size = Math.min(containerWidth - 32, 600);  // -32 for padding
        if (size > 0) {
            gameController.resize(size);
        }
    }
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for debugging
window.gameController = () => gameController;
window.engine = getEngine;
