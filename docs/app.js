document.addEventListener('DOMContentLoaded', () => {
    const predictionsContainer = document.getElementById('predictions-container');
    const filterInput = document.getElementById('filterInput');
    
    // Default colors for teams not in our color map
    const defaultColors = {
        primary: '#cccccc',
        secondary: '#888888'
    };

    const displayStatusMessage = (message) => {
        predictionsContainer.innerHTML = `<div class="status-message">${message}</div>`;
    };

    const loadData = async () => {
        try {
            // Fetch predictions and team colors simultaneously
            const [predictionsResponse, colorsResponse] = await Promise.all([
                fetch('data/predictions.json'),
                fetch('data/team_colors.json').catch(() => ({ ok: false })) // Gracefully handle if colors file is missing
            ]);

            if (!predictionsResponse.ok) {
                throw new Error('Could not load predictions.json. The file might be missing or the GitHub Pages site may still be deploying.');
            }

            const games = await predictionsResponse.json();
            const teamColors = colorsResponse.ok ? await colorsResponse.json() : {};

            if (!games || games.length === 0) {
                displayStatusMessage('No predictions available. Add matchups to <code>docs/input/games.txt</code> and run the workflow.');
                return;
            }

            renderGames(games, teamColors);
            
        } catch (error) {
            console.error('Error loading data:', error);
            displayStatusMessage(`Error: ${error.message}`);
        }
    };

    const renderGames = (games, teamColors) => {
        predictionsContainer.innerHTML = ''; // Clear previous content
        
        games.forEach(game => {
            const awayColor = (teamColors[game.away_team] || defaultColors).primary;
            const homeColor = (teamColors[game.home_team] || defaultColors).primary;

            const homeProbPercent = (game.model_prob_home * 100).toFixed(1);
            const awayProbPercent = (100 - homeProbPercent).toFixed(1);

            const card = document.createElement('div');
            card.className = 'game-card';
            card.innerHTML = `
                <div class="teams">
                    <span class="team-name away">${game.away_team}</span>
                    <span class="vs">vs</span>
                    <span class="team-name home">${game.home_team}</span>
                </div>
                <div class="probability-bar-container">
                    <div class="probability-bar" style="width: ${awayProbPercent}%; background-color: ${awayColor};">${awayProbPercent}%</div>
                    <div class="probability-bar" style="width: ${homeProbPercent}%; background-color: ${homeColor};">${homeProbPercent}%</div>
                </div>
                <div class="pick-container">
                    <span class="pick-label">PICK:</span>
                    <span class="pick-team">${game.pick}</span>
                </div>
            `;
            predictionsContainer.appendChild(card);
        });
    };

    filterInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const gameCards = document.querySelectorAll('.game-card');

        gameCards.forEach(card => {
            const cardText = card.textContent.toLowerCase();
            if (cardText.includes(searchTerm)) {
                card.classList.remove('hidden');
            } else {
                card.classList.add('hidden');
            }
        });
    });

    loadData();
});
