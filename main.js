/**
 * Main entry point for the checkers web app
 */

import { GameController } from './game-controller.js';
import { getEngine } from './engine-api.js';
import { TablebaseLoader, loadNNModelFile } from './tablebase-loader.js';

// Global state
let gameController = null;
let tablebaseLoader = null;

// Edit mode state
let editMode = false;
let editPieceType = 'empty';  // 'empty', 'white-man', 'white-king', 'black-man', 'black-king'
let editWhiteToMove = true;
let editBoard = { white: 0, black: 0, kings: 0 };  // Bitboards for editing

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
            updateUndoRedoButtons();
        };

        gameController.onGameOver = (winner, reason) => {
            showGameOver(winner);
        };

        gameController.onThinkingStart = () => {
            setThinkingIndicator(true);
            updateUndoRedoButtons();
        };

        gameController.onThinkingEnd = () => {
            setThinkingIndicator(false);
            updateUndoRedoButtons();
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
        updateUndoRedoButtons();

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
 * Update undo/redo button states
 */
function updateUndoRedoButtons() {
    if (!gameController) return;

    const undoBtn = document.getElementById('btn-undo');
    const redoBtn = document.getElementById('btn-redo');
    const { canUndo, canRedo } = gameController.getUndoRedoState();

    if (undoBtn) undoBtn.disabled = !canUndo;
    if (redoBtn) redoBtn.disabled = !canRedo;
}

/**
 * Enter edit mode
 */
async function enterEditMode() {
    editMode = true;

    // Copy current position to edit board
    const board = await gameController.getBoard();
    editBoard = { white: board.white, black: board.black, kings: board.kings };
    editWhiteToMove = board.whiteToMove;

    // Update UI
    document.getElementById('game-controls').style.display = 'none';
    document.getElementById('edit-controls').style.display = 'block';
    document.querySelectorAll('.controls-section .control-group:not(#game-controls):not(#edit-controls)').forEach(el => {
        el.style.display = 'none';
    });

    // Update side to move buttons
    updateSideToMoveButtons();

    // Set up edit click handler
    gameController.boardUI.onClick = handleEditClick;
    gameController.boardUI.setLegalMoves([]);  // Clear move highlights
    gameController.boardUI.setSelected(null);
}

/**
 * Exit edit mode and apply position
 */
async function exitEditMode() {
    editMode = false;

    // Apply the edited position
    await gameController.setPosition(editBoard.white, editBoard.black, editBoard.kings, editWhiteToMove);

    // Update UI
    document.getElementById('game-controls').style.display = 'block';
    document.getElementById('edit-controls').style.display = 'none';
    document.querySelectorAll('.controls-section .control-group:not(#game-controls):not(#edit-controls)').forEach(el => {
        el.style.display = 'block';
    });

    // Restore normal click handler (bind to gameController context)
    gameController.boardUI.onClick = (square) => gameController._handleSquareClick.call(gameController, square);

    updateMoveHistory();
    updateUndoRedoButtons();
}

/**
 * Get the piece type at a given square
 */
function getPieceAt(square) {
    const bit = 1 << (square - 1);
    const isWhite = (editBoard.white & bit) !== 0;
    const isBlack = (editBoard.black & bit) !== 0;
    const isKing = (editBoard.kings & bit) !== 0;

    if (isWhite && isKing) return 'white-king';
    if (isWhite) return 'white-man';
    if (isBlack && isKing) return 'black-king';
    if (isBlack) return 'black-man';
    return 'empty';
}

/**
 * Rotate to the next piece type
 */
function nextPieceType(current) {
    const order = ['empty', 'white-man', 'white-king', 'black-man', 'black-king'];
    const idx = order.indexOf(current);
    return order[(idx + 1) % order.length];
}

/**
 * Update the piece selector UI to reflect current selection
 */
function updatePieceSelector() {
    document.querySelectorAll('.piece-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.piece === editPieceType);
    });
}

/**
 * Handle board click in edit mode
 */
function handleEditClick(square) {
    if (!editMode || square < 1 || square > 32) return;

    const bit = 1 << (square - 1);

    // Check if square already has the selected piece type
    const currentPiece = getPieceAt(square);
    if (currentPiece === editPieceType) {
        // Rotate to next piece type
        editPieceType = nextPieceType(editPieceType);
        updatePieceSelector();
    }

    // Remove piece from current position
    editBoard.white &= ~bit;
    editBoard.black &= ~bit;
    editBoard.kings &= ~bit;

    // Add new piece based on selected type
    switch (editPieceType) {
        case 'white-man':
            editBoard.white |= bit;
            break;
        case 'white-king':
            editBoard.white |= bit;
            editBoard.kings |= bit;
            break;
        case 'black-man':
            editBoard.black |= bit;
            break;
        case 'black-king':
            editBoard.black |= bit;
            editBoard.kings |= bit;
            break;
        // 'empty' - already cleared above
    }

    // Update board display
    gameController.boardUI.setPosition(editBoard.white, editBoard.black, editBoard.kings, editWhiteToMove);
}

/**
 * Clear the board in edit mode
 */
function clearEditBoard() {
    editBoard = { white: 0, black: 0, kings: 0 };
    gameController.boardUI.setPosition(0, 0, 0, editWhiteToMove);
}

/**
 * Update side to move toggle buttons
 */
function updateSideToMoveButtons() {
    const whiteBtn = document.getElementById('btn-white-to-move');
    const blackBtn = document.getElementById('btn-black-to-move');

    if (whiteBtn && blackBtn) {
        whiteBtn.classList.toggle('active', editWhiteToMove);
        blackBtn.classList.toggle('active', !editWhiteToMove);
    }
}

/**
 * Show new game dialog
 */
function showNewGameDialog() {
    const dialog = document.getElementById('new-game-dialog');
    if (dialog) {
        dialog.style.display = 'flex';
    }
}

/**
 * Hide new game dialog
 */
function hideNewGameDialog() {
    const dialog = document.getElementById('new-game-dialog');
    if (dialog) {
        dialog.style.display = 'none';
    }
}

/**
 * Start a new game with the player as the specified color
 */
async function startNewGame(playAsWhite) {
    hideNewGameDialog();

    // Start the new game first
    await gameController.newGame();

    // Set the human color after newGame() (which resets it)
    if (playAsWhite) {
        gameController.boardUI.setFlipped(false);
        gameController.setHumanColor('white');  // Engine plays black
    } else {
        gameController.boardUI.setFlipped(true);
        gameController.setHumanColor('black');  // Engine plays white
    }

    updateModeButtons();
    updateUndoRedoButtons();
    updateMoveHistory();
}

/**
 * Set up UI event handlers
 */
function setupEventHandlers() {
    // New game button - show dialog
    const newGameBtn = document.getElementById('btn-new-game');
    if (newGameBtn) {
        newGameBtn.addEventListener('click', showNewGameDialog);
    }

    // New game dialog buttons
    const playWhiteBtn = document.getElementById('btn-play-white');
    const playBlackBtn = document.getElementById('btn-play-black');
    const playRandomBtn = document.getElementById('btn-play-random');

    if (playWhiteBtn) {
        playWhiteBtn.addEventListener('click', () => startNewGame(true));
    }
    if (playBlackBtn) {
        playBlackBtn.addEventListener('click', () => startNewGame(false));
    }
    if (playRandomBtn) {
        playRandomBtn.addEventListener('click', () => startNewGame(Math.random() < 0.5));
    }

    // Close dialog when clicking outside
    const newGameDialog = document.getElementById('new-game-dialog');
    if (newGameDialog) {
        newGameDialog.addEventListener('click', (e) => {
            if (e.target === newGameDialog) {
                hideNewGameDialog();
            }
        });
    }

    // Undo button
    const undoBtn = document.getElementById('btn-undo');
    if (undoBtn) {
        undoBtn.addEventListener('click', async () => {
            await gameController.undo();
            updateMoveHistory();
            updateUndoRedoButtons();
        });
    }

    // Redo button
    const redoBtn = document.getElementById('btn-redo');
    if (redoBtn) {
        redoBtn.addEventListener('click', async () => {
            await gameController.redo();
            updateMoveHistory();
            updateUndoRedoButtons();
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

    // Edit mode button
    const editBtn = document.getElementById('btn-edit');
    if (editBtn) {
        editBtn.addEventListener('click', enterEditMode);
    }

    // Piece selector buttons
    document.querySelectorAll('.piece-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.piece-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            editPieceType = btn.dataset.piece;
        });
    });

    // Clear board button
    const clearBtn = document.getElementById('btn-clear-board');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearEditBoard);
    }

    // Side to move buttons
    const whiteToMoveBtn = document.getElementById('btn-white-to-move');
    const blackToMoveBtn = document.getElementById('btn-black-to-move');
    if (whiteToMoveBtn) {
        whiteToMoveBtn.addEventListener('click', () => {
            editWhiteToMove = true;
            updateSideToMoveButtons();
        });
    }
    if (blackToMoveBtn) {
        blackToMoveBtn.addEventListener('click', () => {
            editWhiteToMove = false;
            updateSideToMoveButtons();
        });
    }

    // Done button
    const doneBtn = document.getElementById('btn-edit-done');
    if (doneBtn) {
        doneBtn.addEventListener('click', exitEditMode);
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
