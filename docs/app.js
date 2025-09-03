document.addEventListener('DOMContentLoaded', () => {
    const predictionsContainer = document.getElementById('predictions-container');
    const filterInput = document.getElementById('filterInput');
    
    const defaultColors = {
        primary: '#cccccc',
        secondary: '#888888'
    };

    const displayStatusMessage = (message) => {
        predictionsContainer.innerHTML = `<div class="status-message">${message}</div>`;
    };

    // NEW: A helper function to make the feature names readable
    const formatFeatureName = (feature) => {
        return feature
            .replace(/_/g, ' ')
            .replace('R5', 'Last 5')
            .replace('ppa', 'PPA')
            .replace('diff', 'Diff')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    const loadData = async () => {
        try {
            const [predictionsResponse, colorsResponse] = await Promise.all([
                fetch('data/predictions.json?cache_bust=' + new Date().getTime()), // Added cache bust
                fetch('data/team_colors.json').catch(() => ({ ok: false }))
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
        predictionsContainer.innerHTML = '';
        
        games.forEach(game => {
            const awayColor = (teamColors[game.away_team] || defaultColors).primary;
            const homeColor = (teamColors[game.home_team] || defaultColors).primary;

            const homeProbPercent = (game.model_prob_home * 100).toFixed(1);
            const awayProbPercent = (100 - homeProbPercent).toFixed(1);

            const card = document.createElement('div');
            card.className = 'game-card';

            // --- IMPROVED: Display logic for home/away vs neutral site ---
            const matchupSeparator = game.neutral_site ? '(N)' : '@';
            
            card.innerHTML = `
                <div class="teams">
                    <span class="team-name away">${game.away_team}</span>
                    <span class="vs">${matchupSeparator}</span>
                    <span class="team-name home">${game.home_team}</span>
                </div>
                <div class="probability-bar-container">
                    <div class="probability-bar" style="background-color: ${awayColor}; width: ${awayProbPercent}%;">${awayProbPercent}%</div>
                    <div class="probability-bar" style="background-color: ${homeColor}; width: ${homeProbPercent}%;">${homeProbPercent}%</div>
                </div>
                <div class="pick-container">
                    <span class="pick-label">PICK:</span>
                    <span class="pick-team">${game.pick}</span>
                </div>
                <div class="explanation-details" style="display: none;"></div>
            `;
            
            // --- NEW: Add click event listener to each card ---
            card.addEventListener('click', () => {
                const detailsDiv = card.querySelector('.explanation-details');
                const isHidden = detailsDiv.style.display === 'none';
                
                // Toggle visibility
                detailsDiv.style.display = isHidden ? 'block' : 'none';

                // Populate the explanation the first time it's opened
                if (isHidden && detailsDiv.innerHTML === '') {
                    renderExplanation(detailsDiv, game.explanation, homeColor, awayColor);
                }
            });

            predictionsContainer.appendChild(card);
        });
    };

    // --- NEW: Function to render the explanation details ---
    const renderExplanation = (element, explanation, homeColor, awayColor) => {
        if (!explanation || explanation.length === 0) {
            element.innerHTML = '<div class="explanation-row">No explanation data available.</div>';
            return;
        }

        let maxVal = Math.max(...explanation.map(e => Math.abs(e.value)));
        
        let html = '<h4>Top Factors</h4>';
        explanation.forEach(item => {
            const barWidth = (Math.abs(item.value) / maxVal) * 100;
            const isPositive = item.value > 0; // Positive SHAP value helps the home team
            const color = isPositive ? homeColor : awayColor;
            const textAlign = isPositive ? 'left' : 'right';

            html += `
                <div class="explanation-row">
                    <span class="feature-name">${formatFeatureName(item.feature)}</span>
                    <div class="feature-bar-container">
                        <div class="feature-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${barWidth}%; background-color: ${color};"></div>
                    </div>
                </div>
            `;
        });
        element.innerHTML = html;
    };

    filterInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        document.querySelectorAll('.game-card').forEach(card => {
            const cardText = card.textContent.toLowerCase();
            card.classList.toggle('hidden', !cardText.includes(searchTerm));
        });
    });

    loadData();
});
