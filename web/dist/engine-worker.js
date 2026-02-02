/**
 * Web Worker for the checkers engine
 * Handles WASM module initialization and search requests
 */

// Import the WASM module
importScripts('./engine.js');

let engine = null;
let board = null;
let isReady = false;

/**
 * Initialize the WASM module
 */
async function init() {
    try {
        engine = await CheckersEngine();
        board = engine.getInitialBoard();
        isReady = true;
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: `Failed to initialize engine: ${err.message}` });
    }
}

/**
 * Load tablebase data into the engine
 */
function loadTablebase(materialKey, data) {
    if (!engine) return;
    engine.loadTablebaseData(materialKey, data);
}

/**
 * Load neural network model
 */
function loadNNModel(data, isDTMModel) {
    if (!engine) return;
    engine.loadNNModel(data, isDTMModel);
}

/**
 * Get legal moves for the current position
 */
function getLegalMoves() {
    if (!engine || !board) return [];

    // getLegalMoves now returns plain JS objects directly
    const moves = engine.getLegalMoves(board);
    return moves;
}

/**
 * Make a move on the board
 */
function makeMove(moveData) {
    if (!engine || !board) return null;

    // makeMove accepts a plain JS object with from_xor_to and captures
    board = engine.makeMove(board, moveData);
    return getBoardState();
}

/**
 * Set board position
 */
function setBoard(white, black, kings, whiteToMove) {
    if (!engine) return;
    board = engine.Board.fromBitboards(white, black, kings, whiteToMove);
}

/**
 * Reset to initial position
 */
function resetBoard() {
    if (!engine) return;
    board = engine.getInitialBoard();
    return getBoardState();
}

/**
 * Get current board state
 */
function getBoardState() {
    if (!board) return null;
    return {
        white: board.getWhite(),
        black: board.getBlack(),
        kings: board.getKings(),
        whiteToMove: board.isWhiteToMove(),
        pieceCount: board.pieceCount()
    };
}

// Current search request ID for progress updates
let currentSearchId = null;

/**
 * Perform search with progress updates
 */
function search(maxDepth, maxNodes, requestId) {
    if (!engine || !board) {
        return { error: 'Engine not ready' };
    }

    console.log('Worker: starting search, depth:', maxDepth, 'nodes:', maxNodes);
    currentSearchId = requestId;

    try {
        // Progress callback - sends updates as search progresses
        const progressCallback = (result) => {
            if (currentSearchId !== null) {
                postMessage({
                    id: currentSearchId,
                    type: 'searchProgress',
                    result: {
                        bestMove: result.best_move,
                        score: result.score,
                        depth: result.depth,
                        nodes: result.nodes,
                        tbHits: result.tb_hits,
                        pv: result.pv || []
                    }
                });
            }
        };

        // Use searchWithCallback if available, otherwise fall back to search
        let result;
        if (engine.searchWithCallback) {
            result = engine.searchWithCallback(board, maxDepth || 20, maxNodes || 0, progressCallback);
        } else {
            result = engine.search(board, maxDepth || 20, maxNodes || 0);
        }
        console.log('Worker: search returned:', result);

        currentSearchId = null;
        return {
            bestMove: result.best_move,
            score: result.score,
            depth: result.depth,
            nodes: result.nodes,
            tbHits: result.tb_hits,
            pv: result.pv || []
        };
    } catch (err) {
        console.error('Worker: search error:', err);
        currentSearchId = null;
        return { error: err.message || 'Search failed' };
    }
}

/**
 * Probe DTM tablebase for current position
 */
function probeDTM() {
    if (!engine || !board) return null;
    const dtm = engine.probeDTM(board);
    return dtm;
}

/**
 * Parse move from notation string
 */
function parseMove(notation) {
    if (!engine || !board) return null;

    // parseMove now returns a plain JS object directly (or null)
    const move = engine.parseMove(board, notation);
    return move;
}

/**
 * Check if tablebases/models are loaded
 */
function getLoadedStatus() {
    if (!engine) return { tablebases: false, nnModel: false, dtmNNModel: false };
    return {
        tablebases: engine.hasTablebases(),
        nnModel: engine.hasNNModel(),
        dtmNNModel: engine.hasDTMNNModel()
    };
}

// Message handler
self.onmessage = function(e) {
    const { id, type, data } = e.data;

    let response = { id, type };

    try {
        switch (type) {
            case 'init':
                // Init is async, will post 'ready' message when done
                init();
                return;

            case 'loadTablebase':
                loadTablebase(data.materialKey, data.data);
                response.success = true;
                break;

            case 'loadNNModel':
                loadNNModel(data.data, data.isDTMModel);
                response.success = true;
                break;

            case 'getLegalMoves':
                response.moves = getLegalMoves();
                break;

            case 'makeMove':
                response.board = makeMove(data);
                break;

            case 'setBoard':
                setBoard(data.white, data.black, data.kings, data.whiteToMove);
                response.board = getBoardState();
                break;

            case 'resetBoard':
                response.board = resetBoard();
                break;

            case 'getBoard':
                response.board = getBoardState();
                break;

            case 'search':
                response.result = search(data.maxDepth, data.maxNodes, id);
                break;

            case 'probeDTM':
                response.dtm = probeDTM();
                break;

            case 'parseMove':
                response.move = parseMove(data.notation);
                break;

            case 'getStatus':
                response.status = getLoadedStatus();
                response.ready = isReady;
                break;

            default:
                response.error = `Unknown message type: ${type}`;
        }
    } catch (err) {
        response.error = err.message;
    }

    postMessage(response);
};

// Start initialization
init();
