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

    const formatFeatureName = (feature) => {
        const nameMap = {
            'elo_home_prob': 'Elo Win Probability',
            'is_postseason': 'Postseason Game',
            'neutral_site': 'Neutral Site Game',
            'rest_diff': 'Rest Advantage',
            'spread_home': 'Betting Spread',
            'travel_away_km': 'Away Team Travel'
        };
        if (nameMap[feature]) {
            return nameMap[feature];
        }

        return feature
            .replace(/_/g, ' ')
            .replace('R5', 'Last 5')
            .replace('diff', 'Advantage')
            .replace('ppa', 'PPA')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    const loadData = async () => {
        try {
            const [predictionsResponse, colorsResponse] = await Promise.all([
                fetch('data/predictions.json?cache_bust=' + new Date().getTime()),
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

            const matchupSeparator = game.neutral_site ? 'vs' : '@';
            
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
            
            card.addEventListener('click', () => {
                const detailsDiv = card.querySelector('.explanation-details');
                const isHidden = detailsDiv.style.display === 'none';
                
                detailsDiv.style.display = isHidden ? 'block' : 'none';

                if (isHidden && detailsDiv.innerHTML === '') {
                    renderExplanation(detailsDiv, game.explanation, homeColor, awayColor, game.home_team, game.away_team);
                }
            });

            predictionsContainer.appendChild(card);
        });
    };
    
    /**
     * Renders the new, meter-style explanation.
     */
    const renderExplanation = (element, explanation, homeColor, awayColor, homeTeam, awayTeam) => {
        if (!explanation || explanation.length === 0) {
            element.innerHTML = '<div class="explanation-row">No explanation data available.</div>';
            return;
        }

        let html = '<h4>Top Factors</h4>';
        
        const maxImpact = explanation.reduce((max, item) => Math.max(max, Math.abs(item.value)), 0);

        explanation.forEach(item => {
            const isPositive = item.value > 0;
            const favoredTeam = isPositive ? homeTeam : awayTeam;
            const color = isPositive ? homeColor : awayColor;
            const relativeImpactPercent = maxImpact > 0 ? (Math.abs(item.value) / maxImpact) * 100 : 0;
            const formattedName = formatFeatureName(item.feature);
            
            html += `
                <div class="factor-row">
                    <div class="factor-label">
                        <span class="factor-name">${formattedName}</span>
                        <span class="factor-impact-text" style="color: ${color};">Favors ${favoredTeam}</span>
                    </div>
                    <div class="factor-meter">
                        <div class="meter-bar" style="width: ${relativeImpactPercent.toFixed(1)}%; background-color: ${color};">
                            ${relativeImpactPercent.toFixed(0)}%
                        </div>
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
