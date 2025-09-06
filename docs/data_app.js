document.addEventListener('DOMContentLoaded', () => {
    const dataContainer = document.getElementById('data-container');

    const STAT_DISPLAY_NAMES = {
        'ppa': 'Avg. PPA',
        'successRate': 'Success Rate',
        'explosiveness': 'Explosiveness',
        'rushingPPA': 'Rush PPA',
        'passingPPA': 'Pass PPA',
        'turnovers': 'Turnovers/Game',
        'totalYards': 'Yards/Game'
    };

    const loadDataSnapshot = async () => {
        try {
            const response = await fetch('data/data_snapshot.json?cache_bust=' + new Date().getTime());
            if (!response.ok) {
                throw new Error('Could not load data snapshot. Run the workflow to generate it.');
            }
            const data = await response.json();
            renderData(data);
        } catch (error) {
            dataContainer.innerHTML = `<div class="status-message">Error: ${error.message}</div>`;
            console.error(error);
        }
    };

    const renderData = (data) => {
        const lastUpdated = new Date(data.last_updated).toLocaleString();

        let recentGamesHtml = '<div class="game-result-grid">';
        data.recent_games.forEach(game => {
            recentGamesHtml += `
                <div class="game-result-item">
                    ${game.away_team} ${game.away_points} - ${game.home_points} ${game.home_team}
                </div>
            `;
        });
        recentGamesHtml += '</div>';

        let teamStatsHtml = '<table class="team-stats-table"><thead><tr><th>Team</th><th>Record</th>';
        Object.values(STAT_DISPLAY_NAMES).forEach(name => {
            teamStatsHtml += `<th>${name}</th>`;
        });
        teamStatsHtml += '</tr></thead><tbody>';

        data.team_stats.forEach(team => {
            teamStatsHtml += `<tr><td>${team.team}</td><td>${team.record}</td>`;
            Object.keys(STAT_SDISPLAY_NAMES).forEach(statKey => {
                const value = team[statKey] ? team[statKey].toFixed(2) : '0.00';
                teamStatsHtml += `<td>${value}</td>`;
            });
            teamStatsHtml += '</tr>';
        });
        teamStatsHtml += '</tbody></table>';

        dataContainer.innerHTML = `
            <p style="text-align: center; color: var(--text-secondary); margin-bottom: 20px;">Last Updated: ${lastUpdated}</p>
            <div class="data-section">
                <h2>Results from Week ${data.most_recent_week}</h2>
                ${recentGamesHtml}
            </div>
            <div class="data-section">
                <h2>Season-to-Date Team Stats</h2>
                ${teamStatsHtml}
            </div>
        `;
    };

    loadDataSnapshot();
});
