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
        await engine.init('./engine-worker.js?v=20260208a');

        // Initialize tablebase loader (for downloading only - loading is now lazy in worker)
        try {
            tablebaseLoader = new TablebaseLoader();
            await tablebaseLoader.init();
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

        // Try to load opening book
        try {
            const bookResponse = await fetch('./opening.cbook');
            if (bookResponse.ok) {
                const bookText = await bookResponse.text();
                await engine.loadOpeningBook(bookText);
                console.log('Opening book loaded');
            }
        } catch (err) {
            console.warn('Could not load opening book:', err);
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

        gameController.onTimeUpdate = (secondsLeft) => {
            updateTimeDisplay(secondsLeft);
        };

        // Set up UI event handlers
        setupEventHandlers();
        updateModeButtons();
        updateUndoRedoButtons();

        // Resize board to fit
        // Board sizing is handled by CSS (width: 100%, max-width: 480px)
        // No JavaScript resize needed

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
    await gameController.abortSearch();
    clearSearchInfo();
    editMode = true;

    // Copy current position to edit board
    const board = await gameController.getBoard();
    editBoard = { white: board.white, black: board.black, kings: board.kings };
    editWhiteToMove = board.whiteToMove;

    // Always reset piece type to 'empty' when entering edit mode
    editPieceType = 'empty';
    updatePieceSelector();

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

    // Apply the edited position (with autoPlay disabled to prevent immediate engine move)
    const savedAutoPlay = gameController.autoPlay;
    gameController.autoPlay = false;
    gameController.secondsLeft = 0;
    gameController._notifyTime();
    await gameController.setPosition(editBoard.white, editBoard.black, editBoard.kings, editWhiteToMove);
    gameController.autoPlay = savedAutoPlay;

    // Set player assignments based on mode:
    // - If in 2-player mode ('both'), stay in 2-player mode
    // - Otherwise, make engine play the side NOT to move, so user can move or trigger engine
    if (gameController.humanColor !== 'both') {
        // Human plays the side to move, engine plays the other side
        gameController.humanColor = editWhiteToMove ? 'white' : 'black';
    }

    // Update UI
    document.getElementById('game-controls').style.display = 'flex';
    document.getElementById('edit-controls').style.display = 'none';
    document.querySelectorAll('.controls-section .control-group:not(#game-controls):not(#edit-controls)').forEach(el => {
        el.style.display = 'block';
    });

    // Restore normal click handler (bind to gameController context)
    gameController.boardUI.onClick = (square) => gameController._handleSquareClick.call(gameController, square);

    updateMoveHistory();
    updateUndoRedoButtons();
    updateModeButtons();
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
 * Show time per move dialog
 */
function showTimeDialog() {
    const dialog = document.getElementById('time-dialog');
    const input = document.getElementById('time-input');
    if (dialog && input && gameController) {
        input.value = gameController.secondsPerMove;
        dialog.style.display = 'flex';
        input.select();
    }
}

/**
 * Hide time per move dialog
 */
function hideTimeDialog() {
    const dialog = document.getElementById('time-dialog');
    if (dialog) dialog.style.display = 'none';
}

/**
 * Apply time per move from dialog input
 */
function applyTimePerMove() {
    const input = document.getElementById('time-input');
    if (!input || !gameController) return;

    let value = parseFloat(input.value);
    if (isNaN(value) || value < 0.1) value = 0.1;

    gameController.setSecondsPerMove(value);
    gameController.secondsLeft = 0;
    gameController._notifyTime();
    updateTimePerMoveLabel(value);
    hideTimeDialog();
}

/**
 * Update the "+Xs/move" label
 */
function updateTimePerMoveLabel(seconds) {
    const el = document.getElementById('btn-time-per-move');
    if (!el) return;
    // Format: remove trailing zeros after decimal, but keep at least one decimal for < 1
    let text;
    if (seconds >= 1 && seconds === Math.floor(seconds)) {
        text = `+${seconds}s/move`;
    } else {
        text = `+${parseFloat(seconds.toFixed(1))}s/move`;
    }
    el.textContent = text;
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
 * @param {string} playAs - 'white', 'black', or 'both'
 */
async function startNewGame(playAs) {
    hideNewGameDialog();

    clearSearchInfo();
    await gameController.newGame();

    // Set the human color and board orientation after newGame() (which resets it)
    if (playAs === 'white') {
        gameController.boardUI.setFlipped(false);
        gameController.setHumanColor('white');  // Engine plays black
    } else if (playAs === 'black') {
        gameController.boardUI.setFlipped(true);
        gameController.setHumanColor('black');  // Engine plays white
    } else {
        // 'both' - 2-player mode, board not flipped
        gameController.boardUI.setFlipped(false);
        gameController.setHumanColor('both');
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
    const playBothBtn = document.getElementById('btn-play-both');

    if (playWhiteBtn) {
        playWhiteBtn.addEventListener('click', () => startNewGame('white'));
    }
    if (playBlackBtn) {
        playBlackBtn.addEventListener('click', () => startNewGame('black'));
    }
    if (playBothBtn) {
        playBothBtn.addEventListener('click', () => startNewGame('both'));
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

    // Time per move - click to open dialog
    const timePerMoveBtn = document.getElementById('btn-time-per-move');
    if (timePerMoveBtn) {
        timePerMoveBtn.addEventListener('click', showTimeDialog);
    }

    // Time dialog
    const timeDialog = document.getElementById('time-dialog');
    const timeInput = document.getElementById('time-input');
    const timeOkBtn = document.getElementById('btn-time-ok');
    const timeCancelBtn = document.getElementById('btn-time-cancel');

    if (timeDialog) {
        timeDialog.addEventListener('click', (e) => {
            if (e.target === timeDialog) hideTimeDialog();
        });
    }
    if (timeOkBtn) {
        timeOkBtn.addEventListener('click', applyTimePerMove);
    }
    if (timeCancelBtn) {
        timeCancelBtn.addEventListener('click', hideTimeDialog);
    }
    if (timeInput) {
        timeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') applyTimePerMove();
            if (e.key === 'Escape') hideTimeDialog();
        });
    }

    // Stop button
    const stopBtn = document.getElementById('btn-stop');
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            gameController.stopSearch();
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

// Tablebases are now loaded lazily by the worker - no need to load them here

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
            statusEl.textContent = 'Complete! Tablebases will be used automatically.';
            setTimeout(() => {
                dialog.style.display = 'none';
                // Reload to pick up the new tablebases
                window.location.reload();
            }, 1500);
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
    // Show/hide search info panel
    const searchInfo = document.getElementById('search-info');
    if (searchInfo) {
        searchInfo.style.display = thinking ? 'block' : 'block';  // Keep visible after search
    }

    // Enable/disable stop button
    const stopBtn = document.getElementById('btn-stop');
    if (stopBtn) {
        stopBtn.disabled = !thinking;
    }
}

/**
 * Format and display engine time left
 */
function updateTimeDisplay(seconds) {
    const el = document.getElementById('engine-time-left');
    if (!el) return;

    if (seconds < 0) seconds = 0;

    let text;
    if (seconds >= 3600) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        text = `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    } else {
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        text = `${String(m).padStart(2, '0')}:${s < 10 ? '0' : ''}${s.toFixed(1)}`;
    }
    el.textContent = text;
}

/**
 * Clear search info display
 */
function clearSearchInfo() {
    const searchInfo = document.getElementById('search-info');
    if (searchInfo) searchInfo.style.display = 'none';
}

/**
 * Update search info display
 */
function updateSearchInfo(info) {
    const searchInfo = document.getElementById('search-info');
    if (!searchInfo) return;

    searchInfo.style.display = 'block';

    const summaryEl = document.getElementById('search-summary');
    const pvEl = document.getElementById('search-pv');

    if (summaryEl) {
        const depth = info.depth || '-';
        const score = info.scoreStr || '-';
        const nodes = info.nodes || 0;
        const nodesStr = nodes.toLocaleString();
        const nps = info.nps || 0;
        let npsStr;
        if (nps >= 1000000) {
            npsStr = (nps / 1000000).toFixed(1) + 'M';
        } else if (nps >= 1000) {
            npsStr = (nps / 1000).toFixed(0) + 'k';
        } else {
            npsStr = nps > 0 ? nps.toString() : '-';
        }
        summaryEl.textContent = `${depth}. ${score} nodes:${nodesStr} nps:${npsStr}`;
    }
    if (pvEl) pvEl.textContent = info.pvStr || '-';
}

/**
 * Resize board to fit container
 */
function resizeBoard() {
    // On mobile, let CSS handle scaling (canvas stays at 480x480, CSS scales it down)
    // This avoids feedback loops where measuring causes shrinking
    if (window.innerWidth <= 768) {
        return;
    }

    const section = document.querySelector('.board-section');
    const canvas = document.getElementById('board');

    if (section && canvas && gameController) {
        // Measure the section width (stable, not affected by canvas size)
        const sectionWidth = section.clientWidth;
        // Account for container padding
        const size = Math.min(sectionWidth - 16, 480);
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
