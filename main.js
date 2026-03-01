/**
 * Main entry point for the checkers web app
 * Wires together: GameState, SearchManager, TurnController, MoveInput, BoardUI
 */

const _v = new URL(import.meta.url).searchParams.get('v') || '';
const _q = _v ? `?v=${_v}` : '';
const { getEngine } = await import(`./engine-api.js${_q}`);
const { GameState } = await import(`./game-state.js${_q}`);
const { SearchManager } = await import(`./search-manager.js${_q}`);
const { TurnController } = await import(`./turn-controller.js${_q}`);
const { MoveInput } = await import(`./move-input.js${_q}`);
const { BoardUI } = await import(`./board-ui.js${_q}`);
const { playMoveSound } = await import(`./sound.js${_q}`);
const { TablebaseLoader, loadNNModelFile } = await import(`./tablebase-loader.js${_q}`);
const { saveGame, getGames, deleteGame, clearGames, computeStats } = await import(`./game-storage.js${_q}`);
const { isSoundEnabled, setSoundEnabled } = await import(`./sound.js${_q}`);

// --- Module instances (set during init) ---
let engine = null;
let gameState = null;
let searchManager = null;
let turnController = null;
let moveInput = null;
let boardUI = null;
let tablebaseLoader = null;

// Edit mode state
let editMode = false;
let editPieceType = 'empty';  // 'empty', 'white-man', 'white-king', 'black-man', 'black-king'
let editWhiteToMove = true;
let editBoard = { white: 0, black: 0, kings: 0 };

// Analysis display state
let showAnalysis = false;

// Match play state
let matchPlayActive = false;
let matchStats = { wins: 0, draws: 0, losses: 0 };
const MATCH_STATS_KEY = 'torquemada-match-stats';
let drawOfferResolve = null;

// Opening book state
let useBook = true;

// Saved options for restoring after match play
let savedPonder = false;
let savedShowAnalysis = false;
let savedUseBook = true;

/**
 * Initialize the application
 */
async function init() {
    const canvas = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const loadingEl = document.getElementById('loading');
    const gameContainerEl = document.getElementById('game-container');

    if (loadingEl) loadingEl.style.display = 'flex';
    if (gameContainerEl) gameContainerEl.style.display = 'none';

    try {
        // Initialize engine
        updateLoadingStatus('Iniciando motor...');
        engine = getEngine();
        await engine.init();

        // Initialize tablebase loader
        try {
            tablebaseLoader = new TablebaseLoader();
            await tablebaseLoader.init();
        } catch (err) {
            console.warn('OPFS not available:', err);
        }

        // Load NN model
        updateLoadingStatus('Cargando red neuronal...');
        try {
            const nnData = await loadNNModelFile('./models/model_006.bin');
            await engine.loadNNModel(nnData);
        } catch (err) {
            console.warn('Could not load NN model:', err);
        }

        // Load opening book
        try {
            const bookResponse = await fetch('./opening-a822ebc72d1b.cbook');
            if (bookResponse.ok) {
                const bookText = await bookResponse.text();
                await engine.loadOpeningBook(bookText);
                console.log('Opening book loaded');
            }
        } catch (err) {
            console.warn('Could not load opening book:', err);
        }

        // Create modules
        updateLoadingStatus('Iniciando partida...');
        boardUI = new BoardUI(canvas);
        gameState = new GameState(engine);
        searchManager = new SearchManager(engine);
        turnController = new TurnController(gameState, searchManager);
        moveInput = new MoveInput();

        // Initialize
        await gameState.init();
        turnController.init();

        // Wire events
        wireGameStateEvents();
        wireSearchManagerEvents();
        wireTurnControllerEvents();
        wireBoardClicks();

        // Restore preferences
        await restoreSavedPreferences();

        // Set up UI event handlers
        setupEventHandlers();
        updateBoardFromState();
        updateModeButtons();
        updateOptionsButtons();
        updateUndoRedoButtons();
        updatePlayButton();

        // Update status
        statusEl.textContent = 'Nueva partida';

        // Hide loading, show game
        if (loadingEl) loadingEl.style.display = 'none';
        if (gameContainerEl) gameContainerEl.style.display = 'flex';

        const status = await engine.getStatus();
        console.log('Engine status:', status);

    } catch (err) {
        console.error('Initialization failed:', err);
        updateLoadingStatus(`Error: ${err.message}`);
    }
}

// --- Event wiring ---

function wireGameStateEvents() {
    gameState.addEventListener('newGame', (e) => {
        updateBoardFromState();
        clearInputState();
        boardUI.setSelected(null);
        boardUI.clearLastMove();
        clearSearchInfo();
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        searchManager.clearPV();
        updateStatus('Nueva partida');
    });

    gameState.addEventListener('move', (e) => {
        const { move, board, gameOver } = e.detail;
        updateBoardFromState();
        clearInputState(false);
        boardUI.setLastMove(move.from, move.to);
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        if (!gameOver) {
            const side = board.whiteToMove ? 'blancas' : 'negras';
            updateStatus(`Mueven ${side}`);
        }
    });

    gameState.addEventListener('undo', (e) => {
        const { board, undoneMove } = e.detail;
        updateBoardFromState();
        clearInputState();
        boardUI.setSelected(null);
        // Restore last move highlight if there's still history
        if (gameState.history.length > 0) {
            const prevMove = gameState.history[gameState.history.length - 1].move;
            boardUI.setLastMove(prevMove.from, prevMove.to);
        }
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        searchManager.clearPV();
        updateStatus('Jugada deshecha');
    });

    gameState.addEventListener('redo', (e) => {
        const { board, move } = e.detail;
        updateBoardFromState();
        clearInputState();
        boardUI.setSelected(null);
        boardUI.setLastMove(move.from, move.to);
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        searchManager.clearPV();
        updateStatus('Jugada rehecha');
    });

    gameState.addEventListener('positionChanged', (e) => {
        updateBoardFromState();
        clearInputState();
        boardUI.setSelected(null);
        boardUI.clearLastMove();
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        updateStatus('Posición establecida');
    });

    gameState.addEventListener('gameLoaded', (e) => {
        updateBoardFromState();
        clearInputState();
        boardUI.setSelected(null);
        boardUI.clearLastMove();
        updateMoveHistory();
        updateUndoRedoButtons();
        updatePlayButton();
        updateStatus('Partida cargada');
    });

    gameState.addEventListener('legalMovesUpdated', (e) => {
        const { moves } = e.detail;
        moveInput.setLegalMoves(moves);
        boardUI.setLegalMoves(moves);
    });

    gameState.addEventListener('gameOver', (e) => {
        const { winner, reason } = e.detail;
        showGameOver(winner, reason);
        updatePlayButton();
    });
}

function wireSearchManagerEvents() {
    searchManager.addEventListener('searchInfo', (e) => {
        updateSearchInfo(e.detail);
        updatePlayButton();
    });

    searchManager.addEventListener('searchStart', (e) => {
        if (e.detail.type === 'think') {
            updateStatus('Motor pensando...');
        }
        updatePlayButton();
        updateUndoRedoButtons();
    });

    searchManager.addEventListener('searchEnd', () => {
        updatePlayButton();
        updateUndoRedoButtons();
    });

    searchManager.addEventListener('timeUpdate', (e) => {
        updateTimeDisplay(e.detail.secondsLeft);
    });
}

function wireTurnControllerEvents() {
    turnController.addEventListener('humanColorChanged', () => {
        updateModeButtons();
        updatePlayButton();
    });

    turnController.addEventListener('drawOffer', (e) => {
        const { resolve } = e.detail;
        drawOfferResolve = resolve;
        document.getElementById('draw-offer-dialog').style.display = 'flex';
    });

    // Engine move: animate + sound + commit, then signal done
    turnController.addEventListener('engineMove', async (e) => {
        const { move, resolve } = e.detail;
        await animateAndMakeMove(move, true);  // always animate engine moves
        resolve();
    });
}

function wireBoardClicks() {
    boardUI.onClick = async (square) => {
        if (editMode) {
            handleEditClick(square);
            return;
        }

        if (gameState.gameOver) return;
        if (searchManager.state === 'thinking') return;
        if (!turnController.isHumanTurn()) return;

        // Stop pondering so the worker is free
        if (searchManager.state === 'pondering') {
            await searchManager.abort();
        }

        const result = moveInput.handleClick(square);

        if (!result) {
            // Click didn't match — clear highlights
            clearInputState();
            return;
        }

        if (result.move) {
            await animateAndMakeMove(result.move, result.animate);
        } else if (result.highlights) {
            applyHighlights(result.highlights);
        }
    };
}

// --- Board / UI helpers ---

function updateBoardFromState() {
    if (!gameState.board) return;
    const b = gameState.board;
    boardUI.setPosition(b.white, b.black, b.kings, b.whiteToMove);
}

function clearInputState(render = true) {
    moveInput.clear();
    boardUI.selectedSquare = null;
    boardUI.partialPath = [];
    boardUI.outOfOrderClicks = [];
    boardUI.flexibleHighlights = [];
    boardUI.lastMove = null;
    if (render) {
        boardUI.setLegalMoves(gameState.legalMoves);
    } else {
        boardUI.legalMoves = gameState.legalMoves;
    }
}

function applyHighlights(highlights) {
    boardUI.legalMoves = [];  // suppress default green highlights
    boardUI.outOfOrderClicks = highlights.selected;
    boardUI.flexibleHighlights = highlights.valid;
    if (highlights.partialPath.length > 0) {
        boardUI.selectedSquare = highlights.partialPath[0];
        boardUI.partialPath = highlights.partialPath;
    } else {
        boardUI.selectedSquare = null;
        boardUI.partialPath = [];
    }
    boardUI.render();
}

/**
 * Animate multi-jump moves step-by-step, then commit the move.
 */
async function animateAndMakeMove(move, animate) {
    if (animate && move.path && move.path.length > 2) {
        for (let i = 1; i < move.path.length; i++) {
            const partialPath = move.path.slice(0, i + 1);
            boardUI.setPartialPath(partialPath);
            boardUI.render();
            playMoveSound();
            await sleep(200);
        }
    } else {
        playMoveSound();
    }
    await gameState.makeMove(move);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updateStatus(message) {
    const el = document.getElementById('status');
    if (el) el.textContent = message;
}

function updateLoadingStatus(message) {
    const el = document.getElementById('loading-status');
    if (el) el.textContent = message;
}

// --- Mode buttons ---

function updateModeButtons() {
    const btnEngineWhite = document.getElementById('btn-engine-white');
    const btnEngineBlack = document.getElementById('btn-engine-black');
    const btnTwoPlayer = document.getElementById('btn-two-player');

    if (!btnEngineWhite || !btnEngineBlack || !btnTwoPlayer) return;

    btnEngineWhite.classList.remove('active');
    btnEngineBlack.classList.remove('active');
    btnTwoPlayer.classList.remove('active');

    switch (turnController.humanColor) {
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

// --- Preferences ---

const PREFS_KEY = 'torquemada-preferences';

function savePreferences() {
    const prefs = {
        ponder: turnController.ponderEnabled,
        showAnalysis,
        useBook,
        sound: isSoundEnabled(),
    };
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

function loadPreferences() {
    try {
        const data = localStorage.getItem(PREFS_KEY);
        return data ? JSON.parse(data) : null;
    } catch {
        return null;
    }
}

async function restoreSavedPreferences() {
    const prefs = loadPreferences();
    if (!prefs) return;

    if (prefs.ponder) await togglePondering(true);
    if (prefs.showAnalysis) toggleShowAnalysis(true);
    if (prefs.useBook === false) {
        useBook = false;
        engine.setUseBook(false);
    }
    setSoundEnabled(prefs.sound !== false);
}

function updateOptionsButtons() {
    const chkPonder = document.getElementById('chk-ponder');
    const chkUseBook = document.getElementById('chk-use-book');
    const chkShowAnalysis = document.getElementById('chk-show-analysis');

    if (chkPonder) chkPonder.checked = turnController.ponderEnabled;
    if (chkUseBook) chkUseBook.checked = useBook;
    if (chkShowAnalysis) chkShowAnalysis.checked = showAnalysis;

    const chkSound = document.getElementById('chk-sound');
    if (chkSound) chkSound.checked = isSoundEnabled();
}

// --- Undo/Redo ---

function updateUndoRedoButtons() {
    if (!gameState) return;
    const undoBtn = document.getElementById('btn-undo');
    const redoBtn = document.getElementById('btn-redo');
    const { canUndo, canRedo } = gameState.getUndoRedoState();
    const thinking = searchManager.state === 'thinking';

    if (undoBtn) undoBtn.disabled = !canUndo || thinking;
    if (redoBtn) redoBtn.disabled = !canRedo || thinking;
}

// --- Edit Mode ---

async function enterEditMode() {
    await searchManager.abort();
    clearSearchInfo();
    editMode = true;

    const board = gameState.board;
    editBoard = { white: board.white, black: board.black, kings: board.kings };
    editWhiteToMove = board.whiteToMove;

    editPieceType = 'empty';
    updatePieceSelector();

    document.getElementById('game-controls').style.display = 'none';
    document.getElementById('edit-controls').style.display = 'block';
    document.querySelectorAll('.controls-section .control-group:not(#game-controls):not(#edit-controls)').forEach(el => {
        el.style.display = 'none';
    });

    updateSideToMoveButtons();
    boardUI.setLegalMoves([]);
    boardUI.setSelected(null);
    updatePlayButton();
}

async function exitEditMode() {
    editMode = false;

    // Temporarily disable auto-play so setPosition doesn't trigger engine
    const savedAutoPlay = turnController.autoPlay;
    turnController.autoPlay = false;
    searchManager.resetTimeBank();
    await gameState.setPosition(editBoard.white, editBoard.black, editBoard.kings, editWhiteToMove);
    turnController.autoPlay = savedAutoPlay;

    // Set player assignments
    if (turnController.humanColor !== 'both') {
        turnController.setHumanColor(editWhiteToMove ? 'white' : 'black');
    }

    document.getElementById('game-controls').style.display = 'flex';
    document.getElementById('edit-controls').style.display = 'none';
    document.querySelectorAll('.controls-section .control-group:not(#game-controls):not(#edit-controls)').forEach(el => {
        el.style.display = 'block';
    });

    updateMoveHistory();
    updateUndoRedoButtons();
    updateModeButtons();
    updatePlayButton();
}

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

function nextPieceType(current) {
    const order = ['empty', 'white-man', 'white-king', 'black-man', 'black-king'];
    const idx = order.indexOf(current);
    return order[(idx + 1) % order.length];
}

function updatePieceSelector() {
    document.querySelectorAll('.piece-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.piece === editPieceType);
    });
}

function handleEditClick(square) {
    if (!editMode || square < 1 || square > 32) return;

    const bit = 1 << (square - 1);

    const currentPiece = getPieceAt(square);
    if (currentPiece === editPieceType) {
        editPieceType = nextPieceType(editPieceType);
        updatePieceSelector();
    }

    editBoard.white &= ~bit;
    editBoard.black &= ~bit;
    editBoard.kings &= ~bit;

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
    }

    boardUI.setPosition(editBoard.white, editBoard.black, editBoard.kings, editWhiteToMove);
}

function clearEditBoard() {
    editBoard = { white: 0, black: 0, kings: 0 };
    boardUI.setPosition(0, 0, 0, editWhiteToMove);
}

function updateSideToMoveButtons() {
    const whiteBtn = document.getElementById('btn-white-to-move');
    const blackBtn = document.getElementById('btn-black-to-move');
    if (whiteBtn && blackBtn) {
        whiteBtn.classList.toggle('active', editWhiteToMove);
        blackBtn.classList.toggle('active', !editWhiteToMove);
    }
}

// --- Time dialog ---

function showTimeDialog() {
    const dialog = document.getElementById('time-dialog');
    const input = document.getElementById('time-input');
    if (dialog && input) {
        input.value = searchManager.secondsPerMove;
        dialog.style.display = 'flex';
        input.select();
    }
}

function hideTimeDialog() {
    const dialog = document.getElementById('time-dialog');
    if (dialog) dialog.style.display = 'none';
}

function applyTimePerMove() {
    const input = document.getElementById('time-input');
    if (!input) return;

    let value = parseFloat(input.value);
    if (isNaN(value) || value < 0.1) value = 0.1;

    searchManager.setSecondsPerMove(value);
    searchManager.resetTimeBank();
    updateTimePerMoveLabel(value);
    hideTimeDialog();
}

function updateTimePerMoveLabel(seconds) {
    const el = document.getElementById('btn-time-per-move');
    if (!el) return;
    let text;
    if (seconds >= 1 && seconds === Math.floor(seconds)) {
        text = `+${seconds}s/mov`;
    } else {
        text = `+${parseFloat(seconds.toFixed(1))}s/mov`;
    }
    el.textContent = text;
}

// --- New game dialog ---

function showNewGameDialog() {
    const dialog = document.getElementById('new-game-dialog');
    if (dialog) dialog.style.display = 'flex';
}

function hideNewGameDialog() {
    const dialog = document.getElementById('new-game-dialog');
    if (dialog) dialog.style.display = 'none';
}

async function startNewGame(playAs) {
    hideNewGameDialog();
    clearSearchInfo();

    await searchManager.abort();
    await gameState.newGame();

    if (playAs === 'white') {
        boardUI.setFlipped(false);
        turnController.setHumanColor('white');
    } else if (playAs === 'black') {
        boardUI.setFlipped(true);
        turnController.setHumanColor('black');
    } else {
        boardUI.setFlipped(false);
        turnController.setHumanColor('both');
    }

    updateModeButtons();
    updateUndoRedoButtons();
    updateMoveHistory();
    updatePlayButton();
}

// --- Play button ---

function updatePlayButton() {
    const playBtn = document.getElementById('btn-play');
    if (!playBtn) return;

    const searchInfo = document.getElementById('search-info');
    if (searchInfo && showAnalysis && searchManager.isSearching) {
        searchInfo.style.display = 'block';
    }

    if (editMode || gameState.gameOver || gameState.legalMoves.length === 0) {
        playBtn.disabled = true;
        return;
    }

    if (searchManager.state === 'pondering') {
        playBtn.disabled = searchManager.currentPV.length === 0;
        return;
    }

    if (searchManager.state === 'thinking') {
        playBtn.disabled = false;
        return;
    }

    // Idle: enabled if it's human's turn
    playBtn.disabled = !turnController.isHumanTurn();
}

// --- Time display ---

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

// --- Search info ---

function clearSearchInfo() {
    const searchInfo = document.getElementById('search-info');
    if (searchInfo) searchInfo.style.display = 'none';
}

function updateSearchInfo(info) {
    const searchInfo = document.getElementById('search-info');
    if (!searchInfo) return;

    if (showAnalysis) searchInfo.style.display = 'block';

    const summaryEl = document.getElementById('search-summary');
    const pvEl = document.getElementById('search-pv');

    if (summaryEl) {
        let depth = info.depth || '-';
        if (info.phase) {
            const phaseLabel = info.phase === 'winning' ? 'ganando' : info.phase === 'losing' ? 'perdiendo' : info.phase;
            depth += ` (${phaseLabel})`;
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

    // Root moves display
    let rootMovesEl = document.getElementById('search-root-moves');
    if (info.rootMoves && info.rootMoves.length > 0) {
        if (!rootMovesEl) {
            rootMovesEl = document.createElement('div');
            rootMovesEl.id = 'search-root-moves';
            rootMovesEl.className = 'search-info-row search-root-moves';
            searchInfo.appendChild(rootMovesEl);
        }
        const parts = info.rootMoves.map(m => {
            let s;
            if (Math.abs(m.score) > 29000) {
                const mateIn = Math.ceil((30000 - Math.abs(m.score)) / 2);
                s = m.score > 0 ? `M${mateIn}` : `-M${mateIn}`;
            } else if (Math.abs(m.score) <= 10000) {
                const val = (m.score / 100).toFixed(2);
                s = (m.score >= 0 ? '+' : '') + val + '=';
            } else {
                const raw = m.score > 0 ? m.score - 10000 : m.score + 10000;
                const val = (raw / 100).toFixed(2);
                s = raw > 0 ? `+${val}` : val;
            }
            return `${m.notation}: ${s}`;
        });
        rootMovesEl.textContent = parts.join('  ');
        rootMovesEl.style.display = '';
    } else if (rootMovesEl) {
        rootMovesEl.style.display = 'none';
    }

    // Book moves display
    let bookMovesEl = document.getElementById('search-book-moves');
    if (info.bookMoves && info.bookMoves.length > 0) {
        if (!bookMovesEl) {
            bookMovesEl = document.createElement('div');
            bookMovesEl.id = 'search-book-moves';
            bookMovesEl.className = 'search-info-row search-book-moves';
            searchInfo.appendChild(bookMovesEl);
        }
        const parts = info.bookMoves.map(m => {
            const pct = (m.probability * 100).toFixed(0);
            const q = (m.q / 100).toFixed(1);
            return `${m.notation}: ${pct}% (${q})`;
        });
        bookMovesEl.innerHTML = `<span class="search-label">libro:</span> ${parts.join('  ')}`;
        bookMovesEl.style.display = '';
    } else if (bookMovesEl) {
        bookMovesEl.style.display = 'none';
    }

    // Update eval bar
    if (info.score !== undefined) {
        updateEvalBar(info.score);
    }
}

// --- Eval bar ---

function normalizeScore(score) {
    if (Math.abs(score) <= 10000) return 0;
    if (Math.abs(score) > 28000) return score > 0 ? 10000 : -10000;
    return score > 0 ? score - 10000 : score + 10000;
}

let lastEvalScore = null;

function updateEvalBar(score) {
    lastEvalScore = score;
    const bar = document.getElementById('eval-bar-white');
    if (!bar) return;

    const flipped = boardUI.flipped;
    const whiteScore = boardUI.whiteToMove ? score : -score;
    const displayScore = normalizeScore(whiteScore);
    const clamped = Math.max(-10000, Math.min(10000, displayScore));
    const pct = ((clamped + 10000) / 20000) * 100;
    bar.style.height = `${pct}%`;

    if (flipped) {
        bar.style.bottom = '';
        bar.style.top = '0';
    } else {
        bar.style.top = '';
        bar.style.bottom = '0';
    }
}

// --- Move history ---

function updateMoveHistory() {
    const historyEl = document.getElementById('move-history');
    if (!historyEl || !gameState) return;

    const { history, redo } = gameState.getMoveHistoryForDisplay();
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

// --- Game over ---

function showGameOver(winner, reason) {
    let message;
    if (winner === 'draw') {
        message = reason ? `¡Tablas — ${reason}!` : '¡Tablas!';
    } else {
        const name = winner === 'white' ? 'blancas' : 'negras';
        message = reason ? `¡Ganan ${name} (${reason})!` : `¡Ganan ${name}!`;
    }

    // Update status text
    if (winner === 'draw') {
        const msg = reason ? `¡Tablas — ${reason}!` : '¡Tablas!';
        updateStatus(msg);
    } else {
        const winnerName = winner === 'white' ? 'blancas' : 'negras';
        updateStatus(`¡Ganan ${winnerName}!`);
    }

    console.log(message);

    if (matchPlayActive) {
        saveGame({
            id: Date.now(),
            date: new Date().toISOString(),
            moves: gameState.history.map(h => h.notation),
            result: winner,
            resultReason: reason || null,
            playerColor: turnController.humanColor
        });

        matchStats = computeStats();

        let title, resultMsg;
        if (winner === 'draw') {
            title = 'Tablas';
            resultMsg = message;
        } else if (winner === turnController.humanColor) {
            title = '¡Ganaste!';
            resultMsg = message;
        } else {
            title = 'Perdiste';
            resultMsg = message;
        }
        setTimeout(() => showMatchResultDialog(title, resultMsg), 100);
    } else {
        setTimeout(() => alert(message), 100);
    }
}

// --- Analysis display ---

function toggleShowAnalysis(enabled) {
    showAnalysis = enabled;

    const evalBar = document.getElementById('eval-bar');
    const searchInfo = document.getElementById('search-info');

    if (enabled) {
        if (evalBar) evalBar.style.visibility = '';
        if (searchInfo && searchManager.isSearching) searchInfo.style.display = 'block';
    } else {
        if (evalBar) evalBar.style.visibility = 'hidden';
        if (searchInfo) searchInfo.style.display = 'none';
    }

    updateOptionsButtons();
    savePreferences();
}

// --- Pondering ---

async function togglePondering(enabled) {
    if (enabled) {
        if (turnController.ponderEnabled) return;
        turnController.setPonderEnabled(true);
        updateOptionsButtons();
        updatePlayButton();
    } else {
        if (!turnController.ponderEnabled) return;
        turnController.setPonderEnabled(false);
        clearSearchInfo();
        updateOptionsButtons();
        updatePlayButton();
    }
    savePreferences();
}

// --- Match Play ---

function getLegacyStats() {
    try {
        const data = localStorage.getItem(MATCH_STATS_KEY);
        if (data) return JSON.parse(data);
    } catch (e) { /* ignore */ }
    return null;
}

async function startMatchPlay() {
    hideNewGameDialog();

    savedPonder = turnController.ponderEnabled;
    savedShowAnalysis = showAnalysis;
    savedUseBook = useBook;

    toggleShowAnalysis(false);

    matchStats = computeStats();
    matchPlayActive = true;
    document.body.classList.add('match-play');
    document.getElementById('game-controls').style.display = 'none';
    document.getElementById('match-toolbar').style.display = 'flex';
    clearSearchInfo();

    let totalGames = matchStats.wins + matchStats.draws + matchStats.losses;
    if (totalGames === 0) {
        const legacy = getLegacyStats();
        if (legacy) {
            totalGames = (legacy.wins || 0) + (legacy.draws || 0) + (legacy.losses || 0);
        }
    }
    const color = (totalGames % 2 === 0) ? 'white' : 'black';

    searchManager.setSecondsPerMove(3);
    useBook = true;
    engine.setUseBook(true);

    await searchManager.abort();
    await gameState.newGame();
    boardUI.setFlipped(color === 'black');
    turnController.setHumanColor(color);

    await togglePondering(true);
}

let resignResolve = null;

function matchResign() {
    document.getElementById('resign-confirm-dialog').style.display = 'flex';
    return new Promise(resolve => {
        resignResolve = resolve;
    }).then(async (confirmed) => {
        if (!confirmed) return;
        await searchManager.abort();

        const engineColor = turnController.humanColor === 'white' ? 'black' : 'white';
        saveGame({
            id: Date.now(),
            date: new Date().toISOString(),
            moves: gameState.history.map(h => h.notation),
            result: engineColor,
            resultReason: 'abandono',
            playerColor: turnController.humanColor
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

    const stats = computeStats();
    const legacy = getLegacyStats();
    const displayStats = (stats.wins + stats.draws + stats.losses > 0) ? stats
        : legacy || { wins: 0, draws: 0, losses: 0 };

    document.getElementById('history-stat-w').textContent = displayStats.wins;
    document.getElementById('history-stat-d').textContent = displayStats.draws;
    document.getElementById('history-stat-l').textContent = displayStats.losses;

    const games = getGames();
    const listEl = document.getElementById('history-list');
    selectedGameId = null;
    updateHistoryButtons();

    if (games.length === 0) {
        listEl.innerHTML = '<div class="history-empty">No hay partidas registradas.</div>';
    } else {
        let html = '';
        for (let i = games.length - 1; i >= 0; i--) {
            const g = games[i];
            const d = new Date(g.date);
            const dateStr = d.toLocaleDateString('es', { month: 'short', day: 'numeric' });
            const timeStr = d.toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' });
            const movesPreview = g.moves.slice(0, 6).join(' ') + (g.moves.length > 6 ? ' ...' : '');

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

    const listEl = document.getElementById('history-list');
    listEl.addEventListener('click', (e) => {
        const item = e.target.closest('.history-item');
        if (!item) return;

        listEl.querySelectorAll('.history-item.selected').forEach(el => el.classList.remove('selected'));
        item.classList.add('selected');
        selectedGameId = parseInt(item.dataset.id);
        updateHistoryButtons();
    });

    document.getElementById('btn-history-close').addEventListener('click', hideHistoryDialog);

    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) hideHistoryDialog();
    });

    document.getElementById('btn-history-delete').addEventListener('click', () => {
        if (!selectedGameId) return;
        deleteGame(selectedGameId);
        selectedGameId = null;
        showHistoryDialog();
    });

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

async function loadGameForAnalysis(game) {
    await togglePondering(false);
    clearSearchInfo();

    const savedAutoPlay = turnController.autoPlay;
    turnController.autoPlay = false;
    await searchManager.abort();
    await gameState.loadGame(game.moves);
    turnController.autoPlay = savedAutoPlay;

    boardUI.setFlipped(game.playerColor === 'black');
    turnController.setHumanColor('both');
    toggleShowAnalysis(true);
    await togglePondering(true);

    updateMoveHistory();
    updateUndoRedoButtons();
    updateModeButtons();
}

async function exitMatchPlay() {
    matchPlayActive = false;
    document.body.classList.remove('match-play');
    document.getElementById('match-toolbar').style.display = 'none';
    document.getElementById('game-controls').style.display = 'flex';
    document.getElementById('match-result-dialog').style.display = 'none';

    await togglePondering(savedPonder);
    toggleShowAnalysis(savedShowAnalysis);
    useBook = savedUseBook;
    engine.setUseBook(useBook);
    updateOptionsButtons();

    updateModeButtons();
    updateUndoRedoButtons();
    updateMoveHistory();
}

// --- Tablebase download ---

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
                window.location.reload();
            }, 1500);
        }
    } catch (err) {
        if (statusEl) statusEl.textContent = `Error: ${err.message}`;
    }
}

// --- Event handlers setup ---

function setupEventHandlers() {
    // New game
    const newGameBtn = document.getElementById('btn-new-game');
    if (newGameBtn) newGameBtn.addEventListener('click', showNewGameDialog);

    const playWhiteBtn = document.getElementById('btn-play-white');
    const playBlackBtn = document.getElementById('btn-play-black');
    const playBothBtn = document.getElementById('btn-play-both');
    if (playWhiteBtn) playWhiteBtn.addEventListener('click', () => startNewGame('white'));
    if (playBlackBtn) playBlackBtn.addEventListener('click', () => startNewGame('black'));
    if (playBothBtn) playBothBtn.addEventListener('click', () => startNewGame('both'));

    const newGameDialog = document.getElementById('new-game-dialog');
    if (newGameDialog) {
        newGameDialog.addEventListener('click', (e) => {
            if (e.target === newGameDialog) hideNewGameDialog();
        });
    }

    // Undo/Redo
    const undoBtn = document.getElementById('btn-undo');
    if (undoBtn) {
        undoBtn.addEventListener('click', async () => {
            await searchManager.abort();
            await gameState.undo();
        });
    }

    const redoBtn = document.getElementById('btn-redo');
    if (redoBtn) {
        redoBtn.addEventListener('click', async () => {
            if (searchManager.state === 'thinking') return;
            if (searchManager.state === 'pondering') await searchManager.abort();
            await gameState.redo();
        });
    }

    // Flip board
    const flipBtn = document.getElementById('btn-flip');
    if (flipBtn) {
        flipBtn.addEventListener('click', () => {
            boardUI.setFlipped(!boardUI.flipped);
            if (lastEvalScore !== null) updateEvalBar(lastEvalScore);
        });
    }

    // Mode toggle buttons
    const btnEngineWhite = document.getElementById('btn-engine-white');
    const btnEngineBlack = document.getElementById('btn-engine-black');
    const btnTwoPlayer = document.getElementById('btn-two-player');

    if (btnEngineWhite) {
        btnEngineWhite.addEventListener('click', async () => {
            if (searchManager.state === 'pondering') await searchManager.abort();
            turnController.setHumanColor('black');
        });
    }
    if (btnEngineBlack) {
        btnEngineBlack.addEventListener('click', async () => {
            if (searchManager.state === 'pondering') await searchManager.abort();
            turnController.setHumanColor('white');
        });
    }
    if (btnTwoPlayer) {
        btnTwoPlayer.addEventListener('click', async () => {
            if (searchManager.state === 'pondering') await searchManager.abort();
            turnController.setHumanColor('both');
        });
    }

    // Pondering checkbox
    const chkPonder = document.getElementById('chk-ponder');
    if (chkPonder) {
        chkPonder.addEventListener('change', () => togglePondering(chkPonder.checked));
    }

    // Time per move
    const timePerMoveBtn = document.getElementById('btn-time-per-move');
    if (timePerMoveBtn) timePerMoveBtn.addEventListener('click', showTimeDialog);

    // Book checkbox
    const chkUseBook = document.getElementById('chk-use-book');
    if (chkUseBook) {
        chkUseBook.addEventListener('change', () => {
            useBook = chkUseBook.checked;
            engine.setUseBook(useBook);
            savePreferences();
        });
    }

    // Analysis display checkbox
    const chkShowAnalysis = document.getElementById('chk-show-analysis');
    if (chkShowAnalysis) {
        chkShowAnalysis.addEventListener('change', () => toggleShowAnalysis(chkShowAnalysis.checked));
    }

    // Sound checkbox
    const chkSound = document.getElementById('chk-sound');
    if (chkSound) {
        chkSound.addEventListener('change', () => {
            setSoundEnabled(chkSound.checked);
            savePreferences();
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
    if (timeOkBtn) timeOkBtn.addEventListener('click', applyTimePerMove);
    if (timeCancelBtn) timeCancelBtn.addEventListener('click', hideTimeDialog);
    if (timeInput) {
        timeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') applyTimePerMove();
            if (e.key === 'Escape') hideTimeDialog();
        });
    }

    // Play button
    const playBtn = document.getElementById('btn-play');
    if (playBtn) {
        playBtn.addEventListener('click', async () => {
            if (searchManager.state === 'pondering') {
                await turnController.playPonderMove();
            } else if (searchManager.state === 'thinking') {
                searchManager.stop();
            } else {
                await turnController.engineMoveNow();
            }
        });
    }

    // Download tablebases
    const downloadBtn = document.getElementById('btn-download-tb');
    if (downloadBtn) downloadBtn.addEventListener('click', () => showDownloadDialog('dtm'));

    const downloadCwdlBtn = document.getElementById('btn-download-cwdl');
    if (downloadCwdlBtn) downloadCwdlBtn.addEventListener('click', () => showDownloadDialog('cwdl'));

    // Edit mode
    const editBtn = document.getElementById('btn-edit');
    if (editBtn) editBtn.addEventListener('click', enterEditMode);

    // Piece selector
    document.querySelectorAll('.piece-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.piece-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            editPieceType = btn.dataset.piece;
        });
    });

    // Clear board
    const clearBtn = document.getElementById('btn-clear-board');
    if (clearBtn) clearBtn.addEventListener('click', clearEditBoard);

    // Side to move
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
    if (doneBtn) doneBtn.addEventListener('click', exitEditMode);

    // Draw offer dialog
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

    // Match play
    const matchPlayBtn = document.getElementById('btn-match-play');
    if (matchPlayBtn) matchPlayBtn.addEventListener('click', startMatchPlay);

    const statsBtn = document.getElementById('btn-stats');
    if (statsBtn) statsBtn.addEventListener('click', showHistoryDialog);

    const matchResignBtn = document.getElementById('btn-match-resign');
    if (matchResignBtn) matchResignBtn.addEventListener('click', matchResign);

    // Resign confirm dialog
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

    // Match result dialog
    const matchResultOkBtn = document.getElementById('btn-match-result-ok');
    if (matchResultOkBtn) matchResultOkBtn.addEventListener('click', exitMatchPlay);

    // History dialog
    setupHistoryDialogHandlers();
}

// --- Board resize ---

function resizeBoard() {
    if (window.innerWidth <= 768) return;

    const section = document.querySelector('.board-section');
    const canvas = document.getElementById('board');

    if (section && canvas && boardUI) {
        const sectionWidth = section.clientWidth;
        const size = Math.min(sectionWidth - 16, 480);
        if (size > 0) {
            boardUI.resize(size);
        }
    }
}

// Start application
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for debugging
window.gameState = () => gameState;
window.searchManager = () => searchManager;
window.turnController = () => turnController;
window.engine = getEngine;
