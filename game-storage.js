/**
 * Game storage - persists completed match-play games to localStorage
 */

const GAMES_KEY = 'torquemada-games';

/**
 * Save a completed game record
 * @param {Object} game - Game data (id, date, moves, result, resultReason, playerColor)
 */
export function saveGame(game) {
    const games = getGames();
    games.push(game);
    try {
        localStorage.setItem(GAMES_KEY, JSON.stringify(games));
    } catch (e) {
        console.warn('Could not save game:', e);
    }
}

/**
 * Get all saved games, most recent first
 * @returns {Array} Array of game objects
 */
export function getGames() {
    try {
        const data = localStorage.getItem(GAMES_KEY);
        if (data) return JSON.parse(data);
    } catch (e) {
        console.warn('Could not load games:', e);
    }
    return [];
}

/**
 * Delete a game by id
 * @param {number} id - Game id (timestamp)
 */
export function deleteGame(id) {
    const games = getGames().filter(g => g.id !== id);
    try {
        localStorage.setItem(GAMES_KEY, JSON.stringify(games));
    } catch (e) {
        console.warn('Could not save games after delete:', e);
    }
}

/**
 * Delete all saved games
 */
export function clearGames() {
    try {
        localStorage.removeItem(GAMES_KEY);
    } catch (e) {
        console.warn('Could not clear games:', e);
    }
}

/**
 * Compute W/D/L stats from stored games
 * @returns {Object} { wins, draws, losses }
 */
export function computeStats() {
    const games = getGames();
    let wins = 0, draws = 0, losses = 0;
    for (const g of games) {
        if (g.result === 'draw') {
            draws++;
        } else if (g.result === g.playerColor) {
            wins++;
        } else {
            losses++;
        }
    }
    return { wins, draws, losses };
}
