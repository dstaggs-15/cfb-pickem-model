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

    /**
     * Translates raw feature names into clean, human-readable labels.
     */
    const formatFeatureName = (feature) => {
        // Specific overrides for clarity
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

        // Generic formatting for statistical differentials
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
     * Renders the new, more intuitive explanation with bars and percentages.
     */
    const renderExplanation = (element, explanation, homeColor, awayColor, homeTeam, awayTeam) => {
        if (!explanation || explanation.length === 0) {
            element.innerHTML = '<div class="explanation-row">No explanation data available.</div>';
            return;
        }

        let html = '<h4>Top Factors</h4>';
        
        // Find the maximum absolute impact value to create a relative scale (0-100%)
        const maxImpact = explanation.reduce((max, item) => Math.max(max, Math.abs(item.value)), 0);

        explanation.forEach(item => {
            const isPositive = item.value > 0; // Positive SHAP value helps the home team
            const favoredTeam = isPositive ? homeTeam : awayTeam;
            const color = isPositive ? homeColor : awayColor;
            
            // Calculate the factor's strength relative to the strongest factor
            const relativeImpactPercent = maxImpact > 0 ? (Math.abs(item.value) / maxImpact) * 100 : 0;

            // NEW: Generate a more descriptive explanation
            let description = '';
            const featureName = item.feature;
            const formattedName = formatFeatureName(featureName);

            if (featureName === 'is_postseason' || featureName === 'neutral_site') {
                description = `${formattedName} status influences prediction`;
            } else if (featureName.includes('prob') || featureName.includes('spread')) {
                description = `${formattedName} favors ${favoredTeam}`;
            } else if (featureName.startsWith('diff')) {
                description = `Recent ${formattedName.replace('Advantage ', '')} favors ${favoredTeam}`;
            } else {
                description = `${formattedName} favors ${favoredTeam}`;
            }
            
            html += `
                <div class="explanation-row">
                    <span class="feature-name">${formattedName}</span>
                    <span class="feature-description">${description}</span>
                </div>
                <div class="impact-bar-row">
                    <div class="impact-bar-container">
                        <div class="impact-bar" style="width: ${relativeImpactPercent.toFixed(1)}%; background-color: ${color};"></div>
                    </div>
                    <span class="impact-percent">${relativeImpactPercent.toFixed(0)}%</span>
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
