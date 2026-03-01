/**
 * Click-to-move translation — extracted from GameController.
 *
 * Maintains a selection mask and partial path. Returns a result object
 * instead of directly making moves. The caller decides what to do:
 *   {move}       → call gameState.makeMove()
 *   {highlights} → update boardUI
 *   null         → click didn't match anything, clear
 */

export class MoveInput {
    constructor() {
        this._selectedMask = 0;   // 32-bit mask of selected squares
        this.partialPath = [];    // For in-order visualization
        this._legalMoves = [];    // Reference to current legal moves
    }

    /**
     * Update legal moves (call when position changes)
     */
    setLegalMoves(moves) {
        this._legalMoves = moves || [];
        this.clear();
    }

    /**
     * Handle a square click.
     * @param {number} square - Square number (1-32)
     * @returns {{move: Object}|{highlights: Object}|null}
     */
    handleClick(square) {
        const bit = (1 << (square - 1)) >>> 0;

        // Already selected this square — try sequential mode (for repeated landing squares)
        if (this._selectedMask & bit) {
            if (this.partialPath.length > 0) {
                const pathMoves = this._filterByPathPrefix(this._getMovesMatchingSelection());
                const nextIdx = this.partialPath.length;
                const extended = pathMoves.filter(m =>
                    m.path && m.path.length > nextIdx && m.path[nextIdx] === square
                );
                if (extended.length > 0) {
                    this.partialPath.push(square);
                    if (this._areAllSameEngineMove(extended)) {
                        const move = extended[0];
                        const animate = move.path && move.path.length > 2 && this.partialPath.length < move.path.length;
                        this.clear();
                        return { move, animate };
                    }
                    return { highlights: this._computeHighlights() };
                }
            }
            return null;
        }

        // Try adding this square to the selection
        const newMask = (this._selectedMask | bit) >>> 0;
        this._selectedMask = newMask;
        let matchingMoves = this._getMovesMatchingSelection();

        if (matchingMoves.length === 0) {
            // No moves match with full set — restart with just this square
            this._selectedMask = bit;
            this.partialPath = [];
            matchingMoves = this._getMovesMatchingSelection();

            if (matchingMoves.length === 0) {
                // This square is in no move at all — clear everything
                this.clear();
                return null;
            }
        }

        // Try to extend partial path for in-order visualization
        this._tryExtendPartialPath(square, matchingMoves);

        // Exactly one engine move (possibly multiple path orderings) → execute
        if (this._areAllSameEngineMove(matchingMoves)) {
            const move = matchingMoves[0];
            const animate = move.path && move.path.length > 2 && this.partialPath.length < move.path.length;
            this.clear();
            return { move, animate };
        }

        // Multiple moves still possible — show highlights and wait
        return { highlights: this._computeHighlights() };
    }

    /**
     * Clear all input state
     */
    clear() {
        this._selectedMask = 0;
        this.partialPath = [];
    }

    // --- Internal ---

    _getMovesMatchingSelection() {
        const sel = this._selectedMask;
        return this._legalMoves.filter(m => ((m.mask & sel) >>> 0) === sel);
    }

    _filterByPathPrefix(moves) {
        if (this.partialPath.length === 0) return moves;
        return moves.filter(m => {
            if (!m.path || m.path.length < this.partialPath.length) return false;
            for (let i = 0; i < this.partialPath.length; i++) {
                if (m.path[i] !== this.partialPath[i]) return false;
            }
            return true;
        });
    }

    _areAllSameEngineMove(moves) {
        if (moves.length <= 1) return true;
        const first = moves[0];
        return moves.every(m =>
            m.from_xor_to === first.from_xor_to && m.captures === first.captures
        );
    }

    _tryExtendPartialPath(square, matchingMoves) {
        if (this.partialPath.length === 0) {
            if (matchingMoves.some(m => m.from === square)) {
                this.partialPath = [square];
            }
            return;
        }
        const nextIdx = this.partialPath.length;
        for (const m of matchingMoves) {
            if (m.path && m.path.length > nextIdx && m.path[nextIdx] === square) {
                this.partialPath.push(square);
                return;
            }
        }
    }

    /**
     * Compute highlight data for the current selection state
     */
    _computeHighlights() {
        let matchingMoves = this._getMovesMatchingSelection();

        // In sequential mode, also filter by path prefix
        if (this.partialPath.length > 0) {
            const prefixFiltered = this._filterByPathPrefix(matchingMoves);
            if (prefixFiltered.length > 0) matchingMoves = prefixFiltered;
        }

        // Collect all unselected squares from matching move masks
        let validMask = 0;
        for (const m of matchingMoves) {
            validMask |= m.mask;
        }
        validMask = (validMask & ~this._selectedMask) >>> 0;

        // In sequential mode, also include next path positions
        if (this.partialPath.length > 0) {
            const nextIdx = this.partialPath.length;
            for (const m of matchingMoves) {
                if (m.path && m.path.length > nextIdx) {
                    validMask |= (1 << (m.path[nextIdx] - 1));
                }
            }
            validMask = validMask >>> 0;
        }

        const validSquares = [];
        for (let i = 0; i < 32; i++) {
            if (validMask & (1 << i)) validSquares.push(i + 1);
        }

        const selectedSquares = [];
        for (let i = 0; i < 32; i++) {
            if (this._selectedMask & (1 << i)) selectedSquares.push(i + 1);
        }

        return {
            selected: selectedSquares,
            valid: validSquares,
            partialPath: this.partialPath.slice()
        };
    }
}
