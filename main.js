/**
 * Main entry point for the checkers web app
 */

const _v = new URL(import.meta.url).searchParams.get('v') || '';
const _q = _v ? `?v=${_v}` : '';
const { GameController } = await import(`./game-controller.js${_q}`);
const { getEngine } = await import(`./engine-api.js${_q}`);
const { TablebaseLoader, loadNNModelFile } = await import(`./tablebase-loader.js${_q}`);
const { saveGame, getGames, deleteGame, clearGames, computeStats } = await import(`./game-storage.js${_q}`);

// Global state
let gameController = null;
let tablebaseLoader = null;

// Edit mode state
let editMode = false;
let editPieceType = 'empty';  // 'empty', 'white-man', 'white-king', 'black-man', 'black-king'
let editWhiteToMove = true;
let editBoard = { white: 0, black: 0, kings: 0 };  // Bitboards for editing

// Match play state
let matchPlayActive = false;
let matchStats = { wins: 0, draws: 0, losses: 0 };
const MATCH_STATS_KEY = 'torquemada-match-stats';
let drawOfferResolve = null;

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
        updateLoadingStatus('Iniciando motor...');
        const engine = getEngine();
        await engine.init();

        // Initialize tablebase loader (for downloading only - loading is now lazy in worker)
        try {
            tablebaseLoader = new TablebaseLoader();
            await tablebaseLoader.init();
        } catch (err) {
            console.warn('OPFS not available:', err);
        }

        // Try to load NN models
        updateLoadingStatus('Cargando red neuronal...');
        try {
            // Try loading from local files first
            const nnData = await loadNNModelFile('./models/model_006.bin');
            await engine.loadNNModel(nnData);
        } catch (err) {
            console.warn('Could not load NN model:', err);
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
        updateLoadingStatus('Iniciando partida...');
        gameController = new GameController(canvas, engine, statusEl);
        await gameController.init();

        // Set up callbacks
        gameController.onMove = (move, board) => {
            updateMoveHistory();
            updateUndoRedoButtons();
        };

        gameController.onGameOver = (winner, reason) => {
            showGameOver(winner, reason);
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

        gameController.onDrawOffer = () => {
            return new Promise(resolve => {
                drawOfferResolve = (accepted) => {
                    if (!accepted) gameController.onDrawOffer = null;
                    resolve(accepted);
                };
                document.getElementById('draw-offer-dialog').style.display = 'flex';
            });
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
    const btnAnalysis = document.getElementById('btn-analysis');

    if (!btnEngineWhite || !btnEngineBlack || !btnTwoPlayer) return;

    // Remove active from all
    btnEngineWhite.classList.remove('active');
    btnEngineBlack.classList.remove('active');
    btnTwoPlayer.classList.remove('active');
    if (btnAnalysis) btnAnalysis.classList.remove('active');

    if (gameController.analysisMode) {
        if (btnAnalysis) btnAnalysis.classList.add('active');
        return;
    }

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
        text = `+${seconds}s/mov`;
    } else {
        text = `+${parseFloat(seconds.toFixed(1))}s/mov`;
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
    exitAnalysisMode();

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

    // Undo button - stops search if engine is thinking, then undoes
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
            exitAnalysisMode();
            gameController.setHumanColor('black'); // human plays black = engine plays white
            updateModeButtons();
        });
    }
    if (btnEngineBlack) {
        btnEngineBlack.addEventListener('click', () => {
            exitAnalysisMode();
            gameController.setHumanColor('white'); // human plays white = engine plays black
            updateModeButtons();
        });
    }
    if (btnTwoPlayer) {
        btnTwoPlayer.addEventListener('click', () => {
            exitAnalysisMode();
            gameController.setHumanColor('both');
            updateModeButtons();
        });
    }

    // Analysis mode button
    const btnAnalysis = document.getElementById('btn-analysis');
    if (btnAnalysis) {
        btnAnalysis.addEventListener('click', () => enterAnalysisMode());
    }

    // Time per move - click to open dialog
    const timePerMoveBtn = document.getElementById('btn-time-per-move');
    if (timePerMoveBtn) {
        timePerMoveBtn.addEventListener('click', showTimeDialog);
    }

    // Book toggle
    const useBookBtn = document.getElementById('btn-use-book');
    if (useBookBtn) {
        useBookBtn.addEventListener('click', () => {
            gameController.setUseBook(!gameController.useBook);
            useBookBtn.textContent = gameController.useBook ? 'Libro: Sí' : 'Libro: No';
        });
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
        downloadBtn.addEventListener('click', () => showDownloadDialog('dtm'));
    }

    // Download CWDL tablebases button
    const downloadCwdlBtn = document.getElementById('btn-download-cwdl');
    if (downloadCwdlBtn) {
        downloadCwdlBtn.addEventListener('click', () => showDownloadDialog('cwdl'));
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

    // Draw offer dialog buttons
    const acceptDrawBtn = document.getElementById('btn-accept-draw');
    const declineDrawBtn = document.getElementById('btn-decline-draw');
    if (acceptDrawBtn) {
        acceptDrawBtn.addEventListener('click', () => {
            document.getElementById('draw-offer-dialog').style.display = 'none';
            if (drawOfferResolve) { drawOfferResolve(true); drawOfferResolve = null; }
        });
    }
    if (declineDrawBtn) {
        declineDrawBtn.addEventListener('click', () => {
            document.getElementById('draw-offer-dialog').style.display = 'none';
            if (drawOfferResolve) { drawOfferResolve(false); drawOfferResolve = null; }
        });
    }

    // Match Play button in new game dialog
    const matchPlayBtn = document.getElementById('btn-match-play');
    if (matchPlayBtn) {
        matchPlayBtn.addEventListener('click', startMatchPlay);
    }

    // Stats/history button (in main toolbar)
    const statsBtn = document.getElementById('btn-stats');
    if (statsBtn) {
        statsBtn.addEventListener('click', showHistoryDialog);
    }

    const matchResignBtn = document.getElementById('btn-match-resign');
    if (matchResignBtn) {
        matchResignBtn.addEventListener('click', matchResign);
    }

    // Resign confirm dialog buttons
    const resignYesBtn = document.getElementById('btn-resign-yes');
    const resignNoBtn = document.getElementById('btn-resign-no');
    const resignDialog = document.getElementById('resign-confirm-dialog');
    if (resignYesBtn) {
        resignYesBtn.addEventListener('click', () => {
            resignDialog.style.display = 'none';
            if (resignResolve) { resignResolve(true); resignResolve = null; }
        });
    }
    if (resignNoBtn) {
        resignNoBtn.addEventListener('click', () => {
            resignDialog.style.display = 'none';
            if (resignResolve) { resignResolve(false); resignResolve = null; }
        });
    }
    if (resignDialog) {
        resignDialog.addEventListener('click', (e) => {
            if (e.target === resignDialog) {
                resignDialog.style.display = 'none';
                if (resignResolve) { resignResolve(false); resignResolve = null; }
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && resignDialog.style.display !== 'none') {
                resignDialog.style.display = 'none';
                if (resignResolve) { resignResolve(false); resignResolve = null; }
            }
        });
    }

    // Match result dialog OK button
    const matchResultOkBtn = document.getElementById('btn-match-result-ok');
    if (matchResultOkBtn) {
        matchResultOkBtn.addEventListener('click', exitMatchPlay);
    }

    // History dialog
    setupHistoryDialogHandlers();
}

// Tablebases are now loaded lazily by the worker - no need to load them here

/**
 * Show tablebase download dialog
 * @param {string} type - 'dtm' for 5-piece DTM, 'cwdl' for 6-7 piece WDL
 */
async function showDownloadDialog(type = 'dtm') {
    const dialog = document.getElementById('download-dialog');
    if (!dialog) {
        alert('Descarga de finales no disponible');
        return;
    }

    if (!tablebaseLoader || !tablebaseLoader.isAvailable()) {
        alert('El almacenamiento de finales requiere OPFS (Origin Private File System), que necesita HTTPS o localhost.');
        return;
    }

    dialog.style.display = 'flex';

    const progressEl = dialog.querySelector('.progress');
    const statusEl = dialog.querySelector('.status');
    const titleEl = dialog.querySelector('h2');
    const cancelBtn = dialog.querySelector('.cancel-btn');

    if (titleEl) {
        titleEl.textContent = type === 'cwdl'
            ? 'Descargando finales WDL 6-7 piezas'
            : 'Descargando finales DTM 5 piezas';
    }

    let cancelled = false;
    cancelBtn.onclick = () => {
        cancelled = true;
        dialog.style.display = 'none';
    };

    try {
        const downloadFn = type === 'cwdl'
            ? tablebaseLoader.downloadCWDLTablebases.bind(tablebaseLoader)
            : tablebaseLoader.downloadTablebases.bind(tablebaseLoader);

        await downloadFn((loaded, total, file) => {
            if (cancelled) return;
            const pct = Math.round((loaded / total) * 100);
            if (progressEl) progressEl.style.width = `${pct}%`;
            if (statusEl) statusEl.textContent = `Descargando: ${file} (${loaded}/${total})`;
        });

        if (!cancelled) {
            statusEl.textContent = '¡Listo! Se usarán automáticamente.';
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
function showGameOver(winner, reason) {
    let message;
    if (winner === 'draw') {
        message = reason ? `¡Tablas — ${reason}!` : '¡Tablas!';
    } else {
        const name = winner === 'white' ? 'blancas' : 'negras';
        message = reason ? `¡Ganan ${name} (${reason})!` : `¡Ganan ${name}!`;
    }
    console.log(message);

    if (matchPlayActive) {
        // Record the game
        saveGame({
            id: Date.now(),
            date: new Date().toISOString(),
            moves: gameController.history.map(h => h.notation),
            result: winner,
            resultReason: reason || null,
            playerColor: gameController.humanColor
        });

        // Recompute stats from stored games
        matchStats = computeStats();

        // Determine result message from player's perspective
        let title, resultMsg;
        if (winner === 'draw') {
            title = 'Tablas';
            resultMsg = message;
        } else if (winner === gameController.humanColor) {
            title = '¡Ganaste!';
            resultMsg = message;
        } else {
            title = 'Perdiste';
            resultMsg = message;
        }
        setTimeout(() => showMatchResultDialog(title, resultMsg), 100);
    } else {
        // Small delay so the board finishes rendering before the dialog appears
        setTimeout(() => alert(message), 100);
    }
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
    if (matchPlayActive && !gameController.analysisMode) return;
    const searchInfo = document.getElementById('search-info');
    if (!searchInfo) return;

    searchInfo.style.display = 'block';

    const summaryEl = document.getElementById('search-summary');
    const pvEl = document.getElementById('search-pv');

    if (summaryEl) {
        let depth = info.depth || '-';
        if (info.phase) {
            depth += ` (${info.phase})`;
        }
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
        summaryEl.innerHTML = `${depth}. ${score} <span class="search-label">nodes:</span> ${nodesStr} <span class="search-label">nps:</span> ${npsStr}`;
    }
    if (pvEl) pvEl.textContent = info.pvStr || '-';

    // Update eval bar in analysis mode
    if (gameController.analysisMode && info.score !== undefined) {
        updateEvalBar(info.score);
    }
}

/**
 * Update the evaluation bar
 * @param {number} score - Raw centipawn score (from side-to-move's perspective)
 * @param {string} scoreStr - Formatted score string for the label
 */
function updateEvalBar(score) {
    const bar = document.getElementById('eval-bar-white');
    if (!bar) return;

    // Convert score to white's perspective (engine score is side-to-move)
    const whiteScore = gameController.boardUI.whiteToMove ? score : -score;

    // Clamp to [-10000, 10000] and linearly interpolate to [0%, 100%]
    const clamped = Math.max(-10000, Math.min(10000, whiteScore));
    const pct = ((clamped + 10000) / 20000) * 100;
    bar.style.height = `${pct}%`;
}

/**
 * Load a recorded game for analysis.
 * Replays all moves, rewinds to start, enters analysis mode.
 */
async function loadGameForAnalysis(game) {
    // Exit any current mode
    exitAnalysisMode();
    clearSearchInfo();

    // Disable auto-play during load
    const savedAutoPlay = gameController.autoPlay;
    gameController.autoPlay = false;

    // Load the game (replays moves, rewinds to start)
    await gameController.loadGame(game.moves);

    // Restore auto-play
    gameController.autoPlay = savedAutoPlay;

    // Orient board from player's perspective
    gameController.boardUI.setFlipped(game.playerColor === 'black');

    // Enter analysis mode
    await enterAnalysisMode();

    // Update UI
    updateMoveHistory();
    updateUndoRedoButtons();
}

// --- Analysis Mode ---

let savedUseBook = true;  // Book setting before entering analysis mode

async function enterAnalysisMode() {
    if (gameController.analysisMode) return;

    // Stop any running search
    await gameController.abortSearch();

    gameController.analysisMode = true;
    updateModeButtons();

    // Disable opening book (we want real engine evaluation)
    savedUseBook = gameController.useBook;
    gameController.setUseBook(false);

    // Hide time/book controls (not relevant in analysis)
    const engineTimeGroup = document.querySelector('.input-group');
    if (engineTimeGroup) engineTimeGroup.style.display = 'none';

    // Show eval bar
    const evalBar = document.getElementById('eval-bar');
    if (evalBar) evalBar.style.display = '';

    // Start analyzing current position
    if (!gameController.gameOver) {
        gameController._analyzePosition();
    }
}

function exitAnalysisMode() {
    if (!gameController.analysisMode) return;
    gameController.analysisMode = false;

    // Restore opening book setting
    gameController.setUseBook(savedUseBook);

    // Show time/book controls again
    const engineTimeGroup = document.querySelector('.input-group');
    if (engineTimeGroup) engineTimeGroup.style.display = 'flex';

    // Hide eval bar
    const evalBar = document.getElementById('eval-bar');
    if (evalBar) evalBar.style.display = 'none';

    clearSearchInfo();
}

// --- Match Play ---

// Migration: old match stats were stored as a simple counter.
// Since we now derive stats from the game list, the old key is no longer written.
// We keep legacy stats for display if no games exist yet.
function getLegacyStats() {
    try {
        const data = localStorage.getItem(MATCH_STATS_KEY);
        if (data) return JSON.parse(data);
    } catch (e) { /* ignore */ }
    return null;
}

async function startMatchPlay() {
    hideNewGameDialog();
    exitAnalysisMode();
    matchStats = computeStats();
    matchPlayActive = true;
    document.body.classList.add('match-play');
    document.getElementById('game-controls').style.display = 'none';
    document.getElementById('match-toolbar').style.display = 'flex';
    clearSearchInfo();

    // Compute player color: alternate based on total games played
    // Include legacy stats for the count if no recorded games exist yet
    let totalGames = matchStats.wins + matchStats.draws + matchStats.losses;
    if (totalGames === 0) {
        const legacy = getLegacyStats();
        if (legacy) {
            totalGames = (legacy.wins || 0) + (legacy.draws || 0) + (legacy.losses || 0);
        }
    }
    const color = (totalGames % 2 === 0) ? 'white' : 'black';

    // Fixed settings for match play
    gameController.setSecondsPerMove(3);
    gameController.setUseBook(true);

    await gameController.newGame();
    gameController.setHumanColor(color);
    gameController.boardUI.setFlipped(color === 'black');
}

let resignResolve = null;

function matchResign() {
    document.getElementById('resign-confirm-dialog').style.display = 'flex';
    return new Promise(resolve => {
        resignResolve = resolve;
    }).then(async (confirmed) => {
        if (!confirmed) return;
        await gameController.abortSearch();

        // Determine winner (the side the engine plays)
        const engineColor = gameController.humanColor === 'white' ? 'black' : 'white';
        saveGame({
            id: Date.now(),
            date: new Date().toISOString(),
            moves: gameController.history.map(h => h.notation),
            result: engineColor,
            resultReason: 'abandono',
            playerColor: gameController.humanColor
        });
        matchStats = computeStats();
        showMatchResultDialog('Abandono', 'Has abandonado la partida.');
    });
}

function showMatchResultDialog(title, message) {
    document.getElementById('match-result-title').textContent = title;
    document.getElementById('match-result-message').textContent = message;
    document.getElementById('match-stat-w').textContent = matchStats.wins;
    document.getElementById('match-stat-d').textContent = matchStats.draws;
    document.getElementById('match-stat-l').textContent = matchStats.losses;
    document.getElementById('match-result-dialog').style.display = 'flex';
}

// --- History Dialog ---

let selectedGameId = null;

function showHistoryDialog() {
    const dialog = document.getElementById('history-dialog');
    if (!dialog) return;

    // Compute stats (include legacy if no games yet)
    const stats = computeStats();
    const legacy = getLegacyStats();
    const displayStats = (stats.wins + stats.draws + stats.losses > 0) ? stats
        : legacy || { wins: 0, draws: 0, losses: 0 };

    document.getElementById('history-stat-w').textContent = displayStats.wins;
    document.getElementById('history-stat-d').textContent = displayStats.draws;
    document.getElementById('history-stat-l').textContent = displayStats.losses;

    // Populate game list
    const games = getGames();
    const listEl = document.getElementById('history-list');
    selectedGameId = null;
    updateHistoryButtons();

    if (games.length === 0) {
        listEl.innerHTML = '<div class="history-empty">No hay partidas registradas.</div>';
    } else {
        // Most recent first
        let html = '';
        for (let i = games.length - 1; i >= 0; i--) {
            const g = games[i];
            const d = new Date(g.date);
            const dateStr = d.toLocaleDateString('es', { month: 'short', day: 'numeric' });
            const timeStr = d.toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' });
            const movesPreview = g.moves.slice(0, 6).join(' ') + (g.moves.length > 6 ? ' ...' : '');

            // Result from player perspective
            let resultLabel, resultClass;
            if (g.result === 'draw') {
                resultLabel = 'T';
                resultClass = 'draw';
            } else if (g.result === g.playerColor) {
                resultLabel = 'V';
                resultClass = 'win';
            } else {
                resultLabel = 'D';
                resultClass = 'loss';
            }

            html += `<div class="history-item" data-id="${g.id}">
                <span class="history-date">${dateStr} ${timeStr}</span>
                <span class="history-color ${g.playerColor}"></span>
                <span class="history-moves">${movesPreview}</span>
                <span class="history-result ${resultClass}">${resultLabel}</span>
            </div>`;
        }
        listEl.innerHTML = html;
    }

    dialog.style.display = 'flex';
}

function hideHistoryDialog() {
    const dialog = document.getElementById('history-dialog');
    if (dialog) dialog.style.display = 'none';
    selectedGameId = null;
}

function updateHistoryButtons() {
    const analyzeBtn = document.getElementById('btn-history-analyze');
    const deleteBtn = document.getElementById('btn-history-delete');
    if (analyzeBtn) analyzeBtn.disabled = !selectedGameId;
    if (deleteBtn) deleteBtn.disabled = !selectedGameId;
}

function setupHistoryDialogHandlers() {
    const dialog = document.getElementById('history-dialog');
    if (!dialog) return;

    // Click on game list items
    const listEl = document.getElementById('history-list');
    listEl.addEventListener('click', (e) => {
        const item = e.target.closest('.history-item');
        if (!item) return;

        // Deselect previous
        listEl.querySelectorAll('.history-item.selected').forEach(el => el.classList.remove('selected'));

        // Select this one
        item.classList.add('selected');
        selectedGameId = parseInt(item.dataset.id);
        updateHistoryButtons();
    });

    // Close button
    document.getElementById('btn-history-close').addEventListener('click', hideHistoryDialog);

    // Click outside to close
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) hideHistoryDialog();
    });

    // Delete button
    document.getElementById('btn-history-delete').addEventListener('click', () => {
        if (!selectedGameId) return;
        deleteGame(selectedGameId);
        selectedGameId = null;
        // Refresh the dialog
        showHistoryDialog();
    });

    // Analyze button (placeholder - will be fully implemented in task 4)
    document.getElementById('btn-history-analyze').addEventListener('click', () => {
        if (!selectedGameId) return;
        const games = getGames();
        const game = games.find(g => g.id === selectedGameId);
        if (game) {
            hideHistoryDialog();
            loadGameForAnalysis(game);
        }
    });
}

function exitMatchPlay() {
    matchPlayActive = false;
    document.body.classList.remove('match-play');
    document.getElementById('match-toolbar').style.display = 'none';
    document.getElementById('game-controls').style.display = 'flex';
    document.getElementById('match-result-dialog').style.display = 'none';

    // Refresh UI state
    updateModeButtons();
    updateUndoRedoButtons();
    updateMoveHistory();
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
