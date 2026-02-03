/**
 * Canvas-based checkers board rendering
 * Uses standard checkers notation (1-32 squares)
 *
 * Board layout matches C++ convention (0-indexed internally):
 *   31  30  29  28      (row 8 - black's back rank, top of screen)
 *     27  26  25  24    (row 7)
 *   23  22  21  20      (row 6)
 *     19  18  17  16    (row 5)
 *   15  14  13  12      (row 4)
 *     11  10  09  08    (row 3)
 *   07  06  05  04      (row 2)
 *     03  02  01  00    (row 1 - white's back rank, bottom of screen)
 *
 * Square 1 (index 0) is at bottom-right (H1)
 * Square 32 (index 31) is at top-left (A8)
 */

// Convert square number (1-32) to canvas row/col
// Returns {row, col} where row 0 is top, col 0 is left (column A)
function squareToRowCol(sq) {
    // sq is 1-32, convert to 0-31 for calculation
    const idx = sq - 1;
    const boardRow = Math.floor(idx / 4);  // 0-7, where 0 is row 1 (bottom)
    const posInRow = idx % 4;              // 0-3, position within the row

    // Canvas row: 0 = top (row 8), 7 = bottom (row 1)
    const canvasRow = 7 - boardRow;

    // Column depends on whether it's an even or odd board row
    // Even board rows (1,3,5,7): dark squares on columns B,D,F,H (1,3,5,7)
    // Odd board rows (2,4,6,8): dark squares on columns A,C,E,G (0,2,4,6)
    let col;
    if (boardRow % 2 === 0) {
        // Even board row: pos 0 = col 7 (H), pos 3 = col 1 (B)
        col = 7 - 2 * posInRow;
    } else {
        // Odd board row: pos 0 = col 6 (G), pos 3 = col 0 (A)
        col = 6 - 2 * posInRow;
    }

    return { row: canvasRow, col };
}

// Convert canvas row/col to square number (1-32), returns 0 if not a dark square
function rowColToSquare(canvasRow, col) {
    // Canvas row to board row
    const boardRow = 7 - canvasRow;  // 0 = row 1, 7 = row 8

    // Check if this is a dark square and get position in row
    let posInRow;
    if (boardRow % 2 === 0) {
        // Even board row: dark squares on columns 1,3,5,7
        if (col % 2 !== 1) return 0;
        posInRow = (7 - col) / 2;
    } else {
        // Odd board row: dark squares on columns 0,2,4,6
        if (col % 2 !== 0) return 0;
        posInRow = (6 - col) / 2;
    }

    if (posInRow < 0 || posInRow > 3) return 0;

    // Square index = boardRow * 4 + posInRow
    const idx = boardRow * 4 + posInRow;
    return idx + 1;  // Convert to 1-32
}

// Find the captured square between two landing squares in a capture move
function getCapturedSquare(fromSq, toSq) {
    const from = squareToRowCol(fromSq);
    const to = squareToRowCol(toSq);

    // The captured piece is at the midpoint
    const midRow = (from.row + to.row) / 2;
    const midCol = (from.col + to.col) / 2;

    // Convert back to square number
    return rowColToSquare(midRow, midCol);
}

export class BoardUI {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.size = Math.min(canvas.width, canvas.height);
        this.squareSize = this.size / 8;

        // Board state (bitboards from engine)
        this.white = 0x00000FFF;  // Initial position
        this.black = 0xFFF00000;
        this.kings = 0;
        this.whiteToMove = true;

        // UI state
        this.selectedSquare = null;  // Currently selected piece (1-32)
        this.legalMoves = [];        // Legal moves from engine
        this.lastMove = null;        // Last move made (for highlighting)
        this.flipped = false;        // Board orientation
        this.partialPath = [];       // Squares in current partial capture path

        // Colors
        this.colors = {
            lightSquare: '#F0D9B5',
            darkSquare: '#B58863',
            whitePiece: '#FFFFFF',
            whitePieceStroke: '#333333',
            blackPiece: '#333333',
            blackPieceStroke: '#000000',
            selected: 'rgba(255, 255, 0, 0.5)',
            legalMove: 'rgba(0, 255, 0, 0.4)',
            lastMove: 'rgba(155, 199, 0, 0.4)',
            partialPath: 'rgba(255, 165, 0, 0.5)',  // Orange for intermediate capture squares
            kingMark: '#FFD700'
        };

        // Event handling
        this.onClick = null;  // Callback: (square) => void

        // Set up click handler
        this.canvas.addEventListener('click', (e) => this._handleClick(e));

        // Initial render
        this.render();
    }

    /**
     * Update board state from engine
     */
    setPosition(white, black, kings, whiteToMove) {
        this.white = white >>> 0;  // Ensure unsigned
        this.black = black >>> 0;
        this.kings = kings >>> 0;
        this.whiteToMove = whiteToMove;
        this.render();
    }

    /**
     * Set legal moves (for highlighting)
     */
    setLegalMoves(moves) {
        this.legalMoves = moves || [];
        this.render();
    }

    /**
     * Set selected square
     */
    setSelected(square) {
        this.selectedSquare = square;
        this.render();
    }

    /**
     * Set last move (for highlighting)
     */
    setLastMove(from, to) {
        this.lastMove = { from, to };
        this.render();
    }

    /**
     * Clear last move highlight
     */
    clearLastMove() {
        this.lastMove = null;
        this.render();
    }

    /**
     * Set partial path for step-by-step capture visualization
     */
    setPartialPath(path) {
        this.partialPath = path || [];
        this.render();
    }

    /**
     * Flip board orientation
     */
    setFlipped(flipped) {
        this.flipped = flipped;
        this.render();
    }

    /**
     * Resize canvas
     */
    resize(size) {
        this.canvas.width = size;
        this.canvas.height = size;
        this.size = size;
        this.squareSize = size / 8;
        this.render();
    }

    /**
     * Main render function
     */
    render() {
        const ctx = this.ctx;
        const sq = this.squareSize;

        // Clear canvas
        ctx.clearRect(0, 0, this.size, this.size);

        // Draw board squares
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const displayRow = this.flipped ? 7 - row : row;
                const displayCol = this.flipped ? 7 - col : col;
                const x = displayCol * sq;
                const y = displayRow * sq;

                // Square color
                const isDark = (row + col) % 2 === 1;
                ctx.fillStyle = isDark ? this.colors.darkSquare : this.colors.lightSquare;
                ctx.fillRect(x, y, sq, sq);
            }
        }

        // Highlight last move
        if (this.lastMove) {
            this._highlightSquare(this.lastMove.from, this.colors.lastMove);
            this._highlightSquare(this.lastMove.to, this.colors.lastMove);
        }

        // Calculate captured squares and moving piece info for partial path visualization
        let capturedSquares = new Set();
        let movingPieceOriginal = null;  // Original square of moving piece
        let movingPieceCurrent = null;   // Current position of moving piece
        let movingPieceIsWhite = false;
        let movingPieceIsKing = false;

        if (this.partialPath && this.partialPath.length > 1) {
            movingPieceOriginal = this.partialPath[0];
            movingPieceCurrent = this.partialPath[this.partialPath.length - 1];

            // Determine piece color and type
            const bit = 1 << (movingPieceOriginal - 1);
            movingPieceIsWhite = (this.white & bit) !== 0;
            movingPieceIsKing = (this.kings & bit) !== 0;

            // Calculate captured squares from the path
            for (let i = 0; i < this.partialPath.length - 1; i++) {
                const captured = getCapturedSquare(this.partialPath[i], this.partialPath[i + 1]);
                if (captured > 0) {
                    capturedSquares.add(captured);
                }
            }

            // Highlight the path squares
            for (let i = 1; i < this.partialPath.length; i++) {
                this._highlightSquare(this.partialPath[i], this.colors.partialPath);
            }
        }

        // Highlight selected square (original position of piece)
        if (this.selectedSquare) {
            this._highlightSquare(this.selectedSquare, this.colors.selected);
        }

        // Highlight legal move destinations (next valid squares to click)
        if (this.partialPath && this.partialPath.length > 0 && this.legalMoves.length > 0) {
            // When in partial path mode, highlight from current position
            const currentPos = this.partialPath[this.partialPath.length - 1];
            const movesFromCurrent = this.legalMoves.filter(m => m.from === currentPos);
            for (const move of movesFromCurrent) {
                this._highlightSquare(move.to, this.colors.legalMove);
            }
        } else if (this.selectedSquare && this.legalMoves.length > 0) {
            const movesFromSelected = this.legalMoves.filter(m => m.from === this.selectedSquare);
            for (const move of movesFromSelected) {
                this._highlightSquare(move.to, this.colors.legalMove);
            }
        }

        // Draw pieces
        for (let sqNum = 1; sqNum <= 32; sqNum++) {
            const bit = 1 << (sqNum - 1);
            const isWhite = (this.white & bit) !== 0;
            const isBlack = (this.black & bit) !== 0;
            const isKing = (this.kings & bit) !== 0;

            if (isWhite || isBlack) {
                // Skip the moving piece at its original position
                if (movingPieceOriginal && sqNum === movingPieceOriginal) {
                    continue;
                }

                // Draw captured pieces as semi-transparent
                if (capturedSquares.has(sqNum)) {
                    this._drawPiece(sqNum, isWhite, isKing, 0.3);
                } else {
                    this._drawPiece(sqNum, isWhite, isKing);
                }
            }
        }

        // Draw the moving piece at its current position
        if (movingPieceCurrent && movingPieceOriginal) {
            this._drawPiece(movingPieceCurrent, movingPieceIsWhite, movingPieceIsKing);
        }

        // Draw square numbers (for debugging/learning)
        // this._drawSquareNumbers();
    }

    /**
     * Highlight a square
     */
    _highlightSquare(square, color) {
        if (!square || square < 1 || square > 32) return;

        const { row, col } = squareToRowCol(square);
        const displayRow = this.flipped ? 7 - row : row;
        const displayCol = this.flipped ? 7 - col : col;
        const x = displayCol * this.squareSize;
        const y = displayRow * this.squareSize;

        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y, this.squareSize, this.squareSize);
    }

    /**
     * Draw a piece
     * @param {number} square - Square number (1-32)
     * @param {boolean} isWhite - True if white piece
     * @param {boolean} isKing - True if king
     * @param {number} opacity - Opacity (0-1), default 1.0
     */
    _drawPiece(square, isWhite, isKing, opacity = 1.0) {
        const { row, col } = squareToRowCol(square);
        const displayRow = this.flipped ? 7 - row : row;
        const displayCol = this.flipped ? 7 - col : col;
        const x = displayCol * this.squareSize + this.squareSize / 2;
        const y = displayRow * this.squareSize + this.squareSize / 2;
        const radius = this.squareSize * 0.4;

        const ctx = this.ctx;
        const prevAlpha = ctx.globalAlpha;
        ctx.globalAlpha = opacity;

        // Draw piece body
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isWhite ? this.colors.whitePiece : this.colors.blackPiece;
        ctx.fill();
        ctx.strokeStyle = isWhite ? this.colors.whitePieceStroke : this.colors.blackPieceStroke;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw king crown
        if (isKing) {
            ctx.beginPath();
            ctx.arc(x, y, radius * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = this.colors.kingMark;
            ctx.fill();
            ctx.strokeStyle = isWhite ? this.colors.whitePieceStroke : '#666';
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.globalAlpha = prevAlpha;
    }

    /**
     * Draw square numbers (for debugging)
     */
    _drawSquareNumbers() {
        const ctx = this.ctx;
        ctx.font = `${this.squareSize * 0.2}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let sq = 1; sq <= 32; sq++) {
            const { row, col } = squareToRowCol(sq);
            const displayRow = this.flipped ? 7 - row : row;
            const displayCol = this.flipped ? 7 - col : col;
            const x = displayCol * this.squareSize + this.squareSize / 2;
            const y = displayRow * this.squareSize + this.squareSize / 2;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.fillText(sq.toString(), x, y);
        }
    }

    /**
     * Handle click events
     */
    _handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Convert to row/col
        let col = Math.floor(x / this.squareSize);
        let row = Math.floor(y / this.squareSize);

        // Handle flipped board
        if (this.flipped) {
            row = 7 - row;
            col = 7 - col;
        }

        // Convert to square number
        const square = rowColToSquare(row, col);

        if (square > 0 && this.onClick) {
            this.onClick(square);
        }
    }

    /**
     * Check if a square has a piece of the current side to move
     */
    hasPieceToMove(square) {
        if (square < 1 || square > 32) return false;
        const bit = 1 << (square - 1);
        if (this.whiteToMove) {
            return (this.white & bit) !== 0;
        } else {
            return (this.black & bit) !== 0;
        }
    }

    /**
     * Get moves from a specific square
     */
    getMovesFromSquare(square) {
        return this.legalMoves.filter(m => m.from === square);
    }

    /**
     * Check if a move to a target square is legal from the selected square
     */
    getMoveToSquare(fromSquare, toSquare) {
        return this.legalMoves.find(m => m.from === fromSquare && m.to === toSquare);
    }
}
