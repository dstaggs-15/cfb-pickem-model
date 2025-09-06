document.addEventListener('DOMContentLoaded', () => {
    const predictionsContainer = document.getElementById('predictions-container');
    const filterInput = document.getElementById('filterInput');

    // --- FIX #1: Team colors are now "baked in" to the JavaScript ---
    const TEAM_COLORS = {
        "Air Force": { "primary": "#003087", "secondary": "#B2B4B2" },
        "Akron": { "primary": "#00285e", "secondary": "#84754E" },
        "Alabama": { "primary": "#9e1b32", "secondary": "#828A8F" },
        "Appalachian State": { "primary": "#222", "secondary": "#ffc425" },
        "Arizona": { "primary": "#0c234b", "secondary": "#cc0033" },
        "Arizona State": { "primary": "#8C1D40", "secondary": "#FFC627" },
        "Arkansas": { "primary": "#9d2235", "secondary": "#000000" },
        "Arkansas State": { "primary": "#ce0e2d", "secondary": "#000000" },
        "Army": { "primary": "#000000", "secondary": "#d4bf91" },
        "Auburn": { "primary": "#0c2340", "secondary": "#e87722" },
        "Ball State": { "primary": "#DA0000", "secondary": "#000000" },
        "Baylor": { "primary": "#004834", "secondary": "#ffb81c" },
        "Boise State": { "primary": "#0033A0", "secondary": "#D64309" },
        "Boston College": { "primary": "#98002e", "secondary": "#BC9B6A" },
        "Bowling Green": { "primary": "#4F2C1D", "secondary": "#FF7300" },
        "Buffalo": { "primary": "#005BBB", "secondary": "#B2B4B2" },
        "BYU": { "primary": "#002E5D", "secondary": "#FFFFFF" },
        "California": { "primary": "#003262", "secondary": "#FDB515" },
        "Central Michigan": { "primary": "#6a0032", "secondary": "#FFC425" },
        "Charlotte": { "primary": "#005035", "secondary": "#AFAFAF" },
        "Cincinnati": { "primary": "#E00122", "secondary": "#000000" },
        "Clemson": { "primary": "#F56600", "secondary": "#522D80" },
        "Coastal Carolina": { "primary": "#006f71", "secondary": "#a2774f" },
        "Colorado": { "primary": "#cfb87c", "secondary": "#000000" },
        "Colorado State": { "primary": "#1E4D2B", "secondary": "#C8C372" },
        "Duke": { "primary": "#0736A4", "secondary": "#FFFFFF" },
        "East Carolina": { "primary": "#592a8a", "secondary": "#f0907b" },
        "Eastern Michigan": { "primary": "#006633", "secondary": "#FFFFFF" },
        "FIU": { "primary": "#081E3F", "secondary": "#B6862C" },
        "Florida": { "primary": "#0021A5", "secondary": "#FA4616" },
        "Florida Atlantic": { "primary": "#003366", "secondary": "#B11234" },
        "Florida State": { "primary": "#782F40", "secondary": "#CEB888" },
        "Fresno State": { "primary": "#db0032", "secondary": "#002e6d" },
        "Georgia": { "primary": "#BA0C2F", "secondary": "#000000" },
        "Georgia Southern": { "primary": "#003976", "secondary": "#BDBDBA" },
        "Georgia State": { "primary": "#003976", "secondary": "#FFFFFF" },
        "Georgia Tech": { "primary": "#00223e", "secondary": "#b39454" },
        "Hawai'i": { "primary": "#024731", "secondary": "#ffffff" },
        "Houston": { "primary": "#C8102E", "secondary": "#FFFFFF" },
        "Illinois": { "primary": "#13294B", "secondary": "#E84A27" },
        "Indiana": { "primary": "#990000", "secondary": "#EEEDEB" },
        "Iowa": { "primary": "#FFCD00", "secondary": "#000000" },
        "Iowa State": { "primary": "#C8102E", "secondary": "#F1BE48" },
        "James Madison": { "primary": "#450084", "secondary": "#CBB677" },
        "Kansas": { "primary": "#0051ba", "secondary": "#e8000d" },
        "Kansas State": { "primary": "#512888", "secondary": "#FFFFFF" },
        "Kent State": { "primary": "#00245d", "secondary": "#f0b510" },
        "Kentucky": { "primary": "#0033A0", "secondary": "#FFFFFF" },
        "Liberty": { "primary": "#002d62", "secondary": "#a50026" },
        "Louisiana": { "primary": "#ce2842", "secondary": "#ffffff" },
        "Louisiana Monroe": { "primary": "#800029", "secondary": "#bd9a5f" },
        "Louisiana Tech": { "primary": "#002d72", "secondary": "#d31145" },
        "Louisville": { "primary": "#AD0000", "secondary": "#000000" },
        "LSU": { "primary": "#461D7C", "secondary": "#FDD023" },
        "Marshall": { "primary": "#00B140", "secondary": "#212121" },
        "Maryland": { "primary": "#E03A3E", "secondary": "#FFD520" },
        "Memphis": { "primary": "#003087", "secondary": "#898D8D" },
        "Miami": { "primary": "#F47321", "secondary": "#005030" },
        "Miami (OH)": { "primary": "#C41230", "secondary": "#000000" },
        "Michigan": { "primary": "#00274c", "secondary": "#ffcb05" },
        "Michigan State": { "primary": "#18453B", "secondary": "#FFFFFF" },
        "Middle Tennessee": { "primary": "#0066CC", "secondary": "#000000" },
        "Minnesota": { "primary": "#7A0019", "secondary": "#FFC72C" },
        "Mississippi State": { "primary": "#660000", "secondary": "#C8C372" },
        "Missouri": { "primary": "#000000", "secondary": "#F1B82D" },
        "Navy": { "primary": "#00205B", "secondary": "#C8B078" },
        "NC State": { "primary": "#CC0000", "secondary": "#000000" },
        "Nebraska": { "primary": "#E41C38", "secondary": "#FFFFFF" },
        "Nevada": { "primary": "#003366", "secondary": "#828A8F" },
        "New Mexico": { "primary": "#ba0c2f", "secondary": "#a7a8aa" },
        "New Mexico State": { "primary": "#891216", "secondary": "#FFFFFF" },
        "North Carolina": { "primary": "#7BAFD4", "secondary": "#FFFFFF" },
        "North Texas": { "primary": "#00853e", "secondary": "#000000" },
        "Northern Illinois": { "primary": "#C41230", "secondary": "#8F8988" },
        "Northwestern": { "primary": "#4E2A84", "secondary": "#FFFFFF" },
        "Notre Dame": { "primary": "#0C2340", "secondary": "#AE9142" },
        "Ohio": { "primary": "#205C30", "secondary": "#FFFFFF" },
        "Ohio State": { "primary": "#BB0000", "secondary": "#666666" },
        "Oklahoma": { "primary": "#841617", "secondary": "#FDF9D8" },
        "Oklahoma State": { "primary": "#FF7300", "secondary": "#000000" },
        "Old Dominion": { "primary": "#00507d", "secondary": "#a1d2e1" },
        "Ole Miss": { "primary": "#002147", "secondary": "#d7112c" },
        "Oregon": { "primary": "#154733", "secondary": "#FEE123" },
        "Oregon State": { "primary": "#DC4405", "secondary": "#000000" },
        "Penn State": { "primary": "#041E42", "secondary": "#FFFFFF" },
        "Pittsburgh": { "primary": "#003594", "secondary": "#FFB81C" },
        "Purdue": { "primary": "#CEB888", "secondary": "#000000" },
        "Rice": { "primary": "#00205B", "secondary": "#B2B4B2" },
        "Rutgers": { "primary": "#CC0033", "secondary": "#000000" },
        "Sam Houston State": { "primary": "#F86500", "secondary": "#002855" },
        "San Diego State": { "primary": "#C41230", "secondary": "#000000" },
        "San José State": { "primary": "#00539C", "secondary": "#FFD200" },
        "SMU": { "primary": "#C41230", "secondary": "#0033A0" },
        "South Alabama": { "primary": "#00205B", "secondary": "#BF0D3E" },
        "South Carolina": { "primary": "#73000a", "secondary": "#000000" },
        "South Florida": { "primary": "#006747", "secondary": "#CFC493" },
        "Southern Mississippi": { "primary": "#FFAB00", "secondary": "#000000" },
        "Stanford": { "primary": "#8C1515", "secondary": "#FFFFFF" },
        "Syracuse": { "primary": "#F76900", "secondary": "#002D56" },
        "TCU": { "primary": "#4D1979", "secondary": "#FFFFFF" },
        "Temple": { "primary": "#9d2235", "secondary": "#FFFFFF" },
        "Tennessee": { "primary": "#FF8200", "secondary": "#FFFFFF" },
        "Texas": { "primary": "#BF5700", "secondary": "#333f48" },
        "Texas A&M": { "primary": "#500000", "secondary": "#FFFFFF" },
        "Texas State": { "primary": "#501214", "secondary": "#84754E" },
        "Texas Tech": { "primary": "#CC0000", "secondary": "#000000" },
        "Toledo": { "primary": "#00539f", "secondary": "#ffc425" },
        "Troy": { "primary": "#98002E", "secondary": "#828A8F" },
        "Tulane": { "primary": "#006747", "secondary": "#86BC25" },
        "Tulsa": { "primary": "#002D56", "secondary": "#C5B783" },
        "UAB": { "primary": "#006442", "secondary": "#FFB81C" },
        "UCF": { "primary": "#000000", "secondary": "#BA9B37" },
        "UCLA": { "primary": "#2D68C4", "secondary": "#F2A900" },
        "UConn": { "primary": "#000E2F", "secondary": "#FFFFFF" },
        "UMass": { "primary": "#971B2F", "secondary": "#B2B4B2" },
        "UNLV": { "primary": "#BF0D3E", "secondary": "#898D8D" },
        "USC": { "primary": "#990000", "secondary": "#FFC72C" },
        "Utah": { "primary": "#CC0000", "secondary": "#828A8F" },
        "Utah State": { "primary": "#00263A", "secondary": "#949FB5" },
        "UTEP": { "primary": "#002449", "secondary": "#FF8200" },
        "UTSA": { "primary": "#002A5C", "secondary": "#F8941D" },
        "Vanderbilt": { "primary": "#AE9142", "secondary": "#000000" },
        "Virginia": { "primary": "#002244", "secondary": "#F84C1E" },
        "Virginia Tech": { "primary": "#630031", "secondary": "#CF4420" },
        "Wake Forest": { "primary": "#AE9142", "secondary": "#000000" },
        "Washington": { "primary": "#4B2E83", "secondary": "#B7A57A" },
        "Washington State": { "primary": "#981e32", "secondary": "#5e6a71" },
        "West Virginia": { "primary": "#002855", "secondary": "#EAAA00" },
        "Western Kentucky": { "primary": "#C62828", "secondary": "#000000" },
        "Western Michigan": { "primary": "#532e1f", "secondary": "#b4a169" },
        "Wisconsin": { "primary": "#C5050C", "secondary": "#FFFFFF" },
        "Wyoming": { "primary": "#492F24", "secondary": "#FFC425" },
    };
    
    const defaultColors = {
        primary: '#555555',
        secondary: '#333333'
    };

    const displayStatusMessage = (message) => {
        predictionsContainer.innerHTML = `<div class="status-message">${message}</div>`;
    };

    const formatFeatureName = (feature) => {
        const nameMap = {
            'elo_home_prob': 'Elo Win Probability',
            'is_postseason': 'Postseason Game Context',
            'neutral_site': 'Game Venue',
            'rest_diff': 'Rest Advantage',
            'spread_home': 'Betting Spread',
            'travel_away_km': 'Away Team Travel',
            'market_home_prob': 'Market Implied Win %',
            'over_under': 'Game Total (O/U)'
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
            const predictionsResponse = await fetch('data/predictions.json?cache_bust=' + new Date().getTime());
            
            if (!predictionsResponse.ok) {
                throw new Error('Could not load predictions.json.');
            }

            const games = await predictionsResponse.json();
            
            if (!games || games.length === 0) {
                displayStatusMessage('No predictions available. Add matchups to <code>docs/input/games.txt</code> and run the workflow.');
                return;
            }

            renderGames(games, TEAM_COLORS);
            
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
            card.dataset.teamNames = `${game.home_team.toLowerCase()} ${game.away_team.toLowerCase()}`;

            const matchupSeparator = game.neutral_site ? 'vs' : '@';
            
            card.innerHTML = `
                <div class="game-summary">
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
                </div>
                <div class="explanation-details"></div>
            `;
            
            card.addEventListener('click', () => {
                card.classList.toggle('expanded');
                const detailsDiv = card.querySelector('.explanation-details');
                
                if (card.classList.contains('expanded') && detailsDiv.innerHTML === '') {
                    renderExplanation(detailsDiv, game.explanation, homeColor, awayColor, game.home_team, game.away_team);
                }
            });

            predictionsContainer.appendChild(card);
        });
    };
    
    const renderExplanation = (element, explanation, homeColor, awayColor, homeTeam, awayTeam) => {
        if (!explanation || explanation.length === 0) {
            element.innerHTML = '<div class="explanation-row">No explanation data available.</div>';
            return;
        }

        let html = '<h4>Model Factors</h4>';
        
        let influentialFactors = explanation.filter(item => Math.abs(item.value) > 1e-6);

        if (influentialFactors.length > 5) {
             // Show all influential factors
        } else if (explanation.length > 0) {
            influentialFactors = explanation.slice(0, 5);
        }

        if (influentialFactors.length === 0) {
            element.innerHTML = '<div class="factor-row">Prediction was based on a combination of factors with no single dominant influence.</div>';
            return;
        }

        const displayMaxImpact = Math.max(...influentialFactors.map(f => Math.abs(f.value)));

        influentialFactors.forEach(item => {
            const isPositive = item.value > 0;
            const favoredTeam = isPositive ? homeTeam : awayTeam;
            const color = isPositive ? homeColor : awayColor;
            const relativeImpactPercent = displayMaxImpact > 0 ? (Math.abs(item.value) / displayMaxImpact) * 100 : 0;
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
            const cardText = card.dataset.teamNames;
            card.classList.toggle('hidden', !cardText.includes(searchTerm));
        });
    });

    loadData();
});
